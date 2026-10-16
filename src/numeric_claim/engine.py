"""Computing claims, comparing them with the lock and judging what documents print."""

import inspect
import math
import numbers
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from numeric_claim import fingerprint
from numeric_claim.budget import (
    BAND,
    Attribution,
    Budget,
    Rung,
    Verdict,
    attribute,
    distance,
    draws,
    verdict,
)
from numeric_claim.claim import ClaimContext, ClaimSpec, ClosedForm, Value
from numeric_claim.config import DEFAULT_ALPHA
from numeric_claim.errors import ClaimError, ConfigError
from numeric_claim.literals import Literal, parse_literal
from numeric_claim.loader import LoadedClaim
from numeric_claim.lockfile import LockEntry
from numeric_claim.montecarlo import Summary, TTest, holm, seeds, summarize, t_test, welch
from numeric_claim.relations import holds
from numeric_claim.scan import Occurrence


@dataclass(frozen=True)
class ClaimOutcome:
    """What ``check`` found for one claim; ``status`` is ok, drift, unlocked or error.

    ``reference`` is the declared reference as text and ``resolved`` its number, present only
    if the check needed it. ``summary`` describes a random claim's replicates.
    """

    name: str
    kind: str
    status: str
    value: Value | None
    locked: Value | None
    source: str
    note: str
    reference: str | None = None
    resolved: Literal | None = None
    budget: Budget | None = None
    attribution: Attribution | None = None
    summary: Summary | None = None


@dataclass(frozen=True)
class LockOutcome:
    """What ``lock`` produced for one claim: an entry, or the reason it has none."""

    name: str
    entry: LockEntry | None
    error: str


@dataclass(frozen=True)
class OccurrenceResult:
    """How one document occurrence fared; ``result`` is pass, fail or a named problem.

    ``compared`` is the claim's value or its error, whichever the occurrence asks for.
    """

    occurrence: Occurrence
    result: str
    value: Value | None
    note: str
    compared: Decimal | None = None
    verdict: Verdict | None = None
    test: TTest | None = None


def _checked(spec: ClaimSpec, raw: object) -> Value:
    if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
        raise ClaimError(f"claim returned {type(raw).__name__}, expected an int or a float")
    value = int(raw) if isinstance(raw, numbers.Integral) else raw
    if isinstance(value, int) and spec.kind == "exact":
        return value
    try:
        number = float(value)
    except OverflowError as exc:
        raise ClaimError("claim returned a number too large for a float") from exc
    if not math.isfinite(number):
        raise ClaimError(f"claim returned a non-finite value: {number!r}")
    return number


def compute(
    loaded: LoadedClaim, context: ClaimContext, settings: Mapping[str, Value] | None = None
) -> Value:
    """Call a claim function, ``settings`` replacing its defaults, and validate what it returns."""
    function = loaded.function
    changed = settings or {}
    try:
        positional = [
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) > 1:
            raise TypeError("a claim function takes no arguments or one ClaimContext")
        raw = function(context, **changed) if positional else function(**changed)
    except (Exception, SystemExit) as exc:
        raise ClaimError(f"{type(exc).__name__}: {exc}") from exc
    return _checked(loaded.spec, raw)


def measure(loaded: LoadedClaim, context: ClaimContext) -> tuple[Value, Budget | None]:
    """Compute a claim and, if it has an error budget, run its probes."""
    value = compute(loaded, context)
    probes = loaded.spec.probes
    if probes is None:
        return value, None

    def probe(settings: Mapping[str, Value]) -> float:
        try:
            return float(compute(loaded, context, settings))
        except ClaimError as error:
            shown = ", ".join(f"{name}={setting!r}" for name, setting in settings.items())
            raise ClaimError(f"probe with {shown}: {error}") from error

    nudged, jittered = draws(loaded.spec.name, probes)
    nudge = statistics.stdev(map(probe, nudged)) if nudged else None
    jitter = statistics.stdev(map(probe, jittered)) if jittered else None
    ladders: dict[str, tuple[Rung, ...]] = {}
    for name, settings in probes.ladders.items():
        floating = isinstance(probes.defaults[name], float)
        ladders[name] = tuple(
            Rung(setting, probe({name: float(setting) if floating else setting}))
            for setting in settings
        )
    return value, Budget(nudge, jitter, ladders)


def sample(loaded: LoadedClaim, context: ClaimContext, replicates: int) -> Summary:
    """Run a random claim's replicates, each with the seed derived for it, and summarize them."""
    values: list[float] = []
    for index, seed in enumerate(seeds(loaded.spec.name, replicates)):
        try:
            values.append(float(compute(loaded, replace(context, seed=seed))))
        except ClaimError as error:
            raise ClaimError(f"replicate {index}: {error}") from error
    return summarize(values)


def _evaluate(
    loaded: LoadedClaim, context: ClaimContext
) -> tuple[Value, Budget | None, Summary | None]:
    replicates = loaded.spec.replicates
    if replicates is None:
        value, budget = measure(loaded, context)
        return value, budget, None
    summary = sample(loaded, context, replicates)
    return summary.mean, None, summary


def resolve(reference: Literal | ClosedForm) -> Literal:
    """The reference as a number; a closed form is evaluated with mpmath and has no half-unit.

    The digits mpmath prints are read like any printed number, so a closed form that does not
    give a finite number is a claim error.
    """
    if isinstance(reference, Literal):
        return reference
    try:
        import mpmath
    except ImportError as exc:
        raise ClaimError("the closed-form reference needs mpmath, which is not installed") from exc
    try:
        with mpmath.workdps(reference.dps):
            value = parse_literal(str(mpmath.mpf(reference.function(mpmath)))).value
    except (Exception, SystemExit) as exc:
        raise ClaimError(f"the closed-form reference failed: {type(exc).__name__}: {exc}") from exc
    return Literal(reference.text, value, Decimal(0), False)


def agrees(spec: ClaimSpec, computed: Value, locked: Value, sigma: float = 0.0) -> bool:
    """Whether a computed value matches a locked one under the claim's kind.

    ``sigma`` is the noise of a budgeted claim, of which 3 sigma are allowed when that exceeds the
    tolerance.
    """
    if spec.kind == "exact":
        return computed == locked
    return abs(computed - locked) <= max(spec.atol, spec.rtol * abs(locked), BAND * sigma)


def _stored(
    spec: ClaimSpec, entry: LockEntry | None, digest: str
) -> tuple[Value, Budget | None, Summary | None] | None:
    """The locked value, budget and summary, if computed under ``digest`` and complete."""
    if entry is None or entry.fingerprint != digest:
        return None
    if spec.replicates is not None:
        summary = entry.summary
        if summary is None or summary.replicates != spec.replicates:
            return None
        return entry.value, None, summary
    if spec.probes is None:
        return entry.value, None, None
    budget = None if entry.budget is None else entry.budget.arranged(spec.probes)
    return None if budget is None else (entry.value, budget, None)


def _drift(
    spec: ClaimSpec,
    entry: LockEntry,
    value: Value,
    budget: Budget | None,
    summary: Summary | None,
    alpha: float,
) -> str:
    """Why a computed value disagrees with its lock entry, or nothing if it agrees."""
    if summary is not None:
        if entry.summary is None:
            return f"computed a random claim but locked a {entry.kind} one"
        p = welch(summary, entry.summary)
        if p <= alpha:
            return f"computed mean {value!r} but locked {entry.value!r} (Welch p={p:.3g})"
        return ""
    sigma = 0.0
    if budget is not None:
        sigma = max(budget.sigma, 0.0 if entry.budget is None else entry.budget.sigma)
    if not agrees(spec, value, entry.value, sigma):
        return f"computed {value!r} but locked {entry.value!r}"
    return ""


def check_claim(
    loaded: LoadedClaim,
    context: ClaimContext,
    distributions: Sequence[str],
    entry: LockEntry | None,
    *,
    recompute: bool,
    error_bound: bool = False,
    alpha: float = DEFAULT_ALPHA,
) -> ClaimOutcome:
    """Use the locked value while the fingerprint matches; otherwise compute and compare.

    The reference is resolved only if the claim has an error budget or, as ``error_bound``
    tells, a document compares its error. A recomputed random claim drifts when Welch's test
    separates its mean from the locked one at ``alpha``.
    """
    spec = loaded.spec
    locked = entry.value if entry is not None else None
    resolved: Literal | None = None

    def outcome(
        status: str,
        value: Value | None,
        source: str,
        note: str,
        budget: Budget | None = None,
        summary: Summary | None = None,
    ) -> ClaimOutcome:
        reference = None if spec.reference is None else spec.reference.text
        attribution = None
        if value is not None and budget is not None and resolved is not None:
            attribution = attribute(value, budget, resolved)
        return ClaimOutcome(
            spec.name,
            spec.kind,
            status,
            value,
            locked,
            source,
            note,
            reference,
            resolved,
            budget,
            attribution,
            summary,
        )

    try:
        if spec.reference is not None and (spec.probes is not None or error_bound):
            resolved = resolve(spec.reference)
        current = fingerprint.compute(spec, loaded.module, context, distributions)
        stored = _stored(spec, entry, current.digest)
        if stored is not None and not recompute:
            return outcome("ok", stored[0], "lock", "", stored[1], stored[2])
        value, budget, summary = _evaluate(loaded, context)
    except ClaimError as error:
        return outcome("error", None, "-", str(error))
    if entry is None:
        note = "no lock entry; run `numeric-claim lock`"
        return outcome("unlocked", value, "computed", note, budget, summary)
    note = _drift(spec, entry, value, budget, summary, alpha)
    if note:
        return outcome("drift", value, "computed", note, budget, summary)
    if entry.fingerprint != current.digest:
        note = "fingerprint changed; run `numeric-claim lock` to update the lock"
    elif stored is None:
        held = "a different number of replicates" if summary is not None else "no error budget"
        note = f"the lock entry has {held}; run `numeric-claim lock` to update the lock"
    return outcome("ok", value, "computed", note, budget, summary)


def lock_claim(
    loaded: LoadedClaim, context: ClaimContext, distributions: Sequence[str]
) -> LockOutcome:
    """Fingerprint and compute a claim, with its error budget or replicates, for the lock file."""
    spec = loaded.spec
    try:
        current = fingerprint.compute(spec, loaded.module, context, distributions)
        value, budget, summary = _evaluate(loaded, context)
    except ClaimError as error:
        return LockOutcome(spec.name, None, str(error))
    entry = LockEntry(
        spec.kind, value, current.digest, current.environment, spec.rtol, spec.atol, budget, summary
    )
    return LockOutcome(spec.name, entry, "")


def _judge(
    occurrence: Occurrence,
    spec: ClaimSpec | None,
    outcome: ClaimOutcome | None,
    test: TTest | None,
) -> OccurrenceResult:
    if spec is None or outcome is None:
        note = f"no claim named {occurrence.claim!r}"
        return OccurrenceResult(occurrence, "unknown-claim", None, note)
    if outcome.value is None:
        return OccurrenceResult(occurrence, "claim-error", None, outcome.note)
    compared, half_unit = Decimal(outcome.value), None
    if occurrence.quantity == "error":
        if outcome.resolved is None:
            where = f"{occurrence.file}:{occurrence.line}"
            raise ConfigError(f"{where}: the error of claim {spec.name!r} needs a reference")
        compared = distance(outcome.value, outcome.resolved.value)
        half_unit = outcome.resolved.half_unit
    try:
        literal = parse_literal(occurrence.literal)
    except ValueError as exc:
        return OccurrenceResult(occurrence, "bad-literal", outcome.value, str(exc), compared)
    if test is not None:
        result = "pass" if test.label == "supported" else "fail"
        return OccurrenceResult(occurrence, result, outcome.value, "", compared, None, test)
    tolerance = spec.tolerance(outcome.value)
    if outcome.budget is None:
        passed = holds(occurrence.relation, compared, literal, tolerance)
        result = "pass" if passed else "fail"
        return OccurrenceResult(occurrence, result, outcome.value, "", compared)
    found = verdict(
        occurrence.relation, compared, literal, tolerance, outcome.budget.sigma, half_unit
    )
    result = "pass" if found.label == "supported" else "fail"
    return OccurrenceResult(occurrence, result, outcome.value, "", compared, found)


def _t_test(
    occurrence: Occurrence, spec: ClaimSpec | None, outcome: ClaimOutcome | None
) -> tuple[float, float] | None:
    """t and the p-value of an occurrence of a random claim, if it can be tested."""
    if spec is None or outcome is None or outcome.summary is None:
        return None
    try:
        literal = parse_literal(occurrence.literal)
    except ValueError:
        return None
    summary = outcome.summary
    return t_test(occurrence.relation, literal, summary, spec.tolerance(summary.mean))


def judge_all(
    occurrences: Sequence[Occurrence],
    claims: Sequence[LoadedClaim],
    outcomes: Sequence[ClaimOutcome],
    alpha: float = DEFAULT_ALPHA,
) -> list[OccurrenceResult]:
    """Judge every occurrence against the value used for its claim.

    The p-values of the occurrences of random claims are adjusted together by Holm's procedure;
    an adjusted p-value of at most ``alpha`` makes the printed number violated.
    """
    specs = {loaded.spec.name: loaded.spec for loaded in claims}
    by_name = {outcome.name: outcome for outcome in outcomes}
    judged = [
        (occurrence, specs.get(occurrence.claim), by_name.get(occurrence.claim))
        for occurrence in occurrences
    ]
    raw = [_t_test(*item) for item in judged]
    adjusted = iter(holm([found[1] for found in raw if found is not None]))
    results: list[OccurrenceResult] = []
    for (occurrence, spec, outcome), found in zip(judged, raw, strict=True):
        test = None
        if found is not None:
            p_holm = next(adjusted)
            test = TTest(*found, p_holm, "violated" if p_holm <= alpha else "supported")
        results.append(_judge(occurrence, spec, outcome, test))
    return results
