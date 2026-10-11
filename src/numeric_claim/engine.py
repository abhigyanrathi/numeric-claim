"""Computing claims, comparing them with the lock and judging what documents print."""

import inspect
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

from numeric_claim import fingerprint
from numeric_claim.claim import ClaimContext, ClaimSpec, Value
from numeric_claim.errors import ClaimError
from numeric_claim.literals import parse_literal
from numeric_claim.loader import LoadedClaim
from numeric_claim.lockfile import LockEntry
from numeric_claim.relations import holds
from numeric_claim.scan import Occurrence


@dataclass(frozen=True)
class ClaimOutcome:
    """What ``check`` found for one claim; ``status`` is ok, drift, unlocked or error."""

    name: str
    kind: str
    status: str
    value: Value | None
    locked: Value | None
    source: str
    note: str


@dataclass(frozen=True)
class LockOutcome:
    """What ``lock`` produced for one claim: an entry, or the reason it has none."""

    name: str
    entry: LockEntry | None
    error: str


@dataclass(frozen=True)
class OccurrenceResult:
    """How one document occurrence fared; ``result`` is pass, fail or a named problem."""

    occurrence: Occurrence
    result: str
    value: Value | None
    note: str


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


def compute(loaded: LoadedClaim, context: ClaimContext) -> Value:
    """Call a claim function and validate what it returns."""
    function = loaded.function
    try:
        positional = [
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) > 1:
            raise TypeError("a claim function takes no arguments or one ClaimContext")
        raw = function(context) if positional else function()
    except (Exception, SystemExit) as exc:
        raise ClaimError(f"{type(exc).__name__}: {exc}") from exc
    return _checked(loaded.spec, raw)


def agrees(spec: ClaimSpec, computed: Value, locked: Value) -> bool:
    """Whether a computed value matches a locked one under the claim's kind."""
    if spec.kind == "exact":
        return computed == locked
    return abs(computed - locked) <= max(spec.atol, spec.rtol * abs(locked))


def check_claim(
    loaded: LoadedClaim,
    context: ClaimContext,
    distributions: Sequence[str],
    entry: LockEntry | None,
    *,
    recompute: bool,
) -> ClaimOutcome:
    """Use the locked value while the fingerprint matches; otherwise compute and compare."""
    spec = loaded.spec
    locked = entry.value if entry is not None else None
    try:
        current = fingerprint.compute(spec, loaded.module, context, distributions)
        if entry is not None and entry.fingerprint == current.digest and not recompute:
            return ClaimOutcome(spec.name, spec.kind, "ok", entry.value, locked, "lock", "")
        value = compute(loaded, context)
    except ClaimError as error:
        return ClaimOutcome(spec.name, spec.kind, "error", None, locked, "-", str(error))
    if entry is None:
        note = "no lock entry; run `numeric-claim lock`"
        return ClaimOutcome(spec.name, spec.kind, "unlocked", value, None, "computed", note)
    if not agrees(spec, value, entry.value):
        note = f"computed {value!r} but locked {entry.value!r}"
        return ClaimOutcome(spec.name, spec.kind, "drift", value, locked, "computed", note)
    note = ""
    if entry.fingerprint != current.digest:
        note = "fingerprint changed; run `numeric-claim lock` to update the lock"
    return ClaimOutcome(spec.name, spec.kind, "ok", value, locked, "computed", note)


def lock_claim(
    loaded: LoadedClaim, context: ClaimContext, distributions: Sequence[str]
) -> LockOutcome:
    """Fingerprint and compute a claim for the lock file."""
    spec = loaded.spec
    try:
        current = fingerprint.compute(spec, loaded.module, context, distributions)
        value = compute(loaded, context)
    except ClaimError as error:
        return LockOutcome(spec.name, None, str(error))
    entry = LockEntry(spec.kind, value, current.digest, current.environment, spec.rtol, spec.atol)
    return LockOutcome(spec.name, entry, "")


def _judge(
    occurrence: Occurrence, spec: ClaimSpec | None, outcome: ClaimOutcome | None
) -> OccurrenceResult:
    if spec is None or outcome is None:
        note = f"no claim named {occurrence.claim!r}"
        return OccurrenceResult(occurrence, "unknown-claim", None, note)
    if outcome.value is None:
        return OccurrenceResult(occurrence, "claim-error", None, outcome.note)
    try:
        literal = parse_literal(occurrence.literal)
    except ValueError as exc:
        return OccurrenceResult(occurrence, "bad-literal", outcome.value, str(exc))
    passed = holds(occurrence.relation, outcome.value, literal, spec.tolerance(outcome.value))
    return OccurrenceResult(occurrence, "pass" if passed else "fail", outcome.value, "")


def judge_all(
    occurrences: Sequence[Occurrence],
    claims: Sequence[LoadedClaim],
    outcomes: Sequence[ClaimOutcome],
) -> list[OccurrenceResult]:
    """Judge every occurrence against the value used for its claim."""
    specs = {loaded.spec.name: loaded.spec for loaded in claims}
    by_name = {outcome.name: outcome for outcome in outcomes}
    return [
        _judge(occurrence, specs.get(occurrence.claim), by_name.get(occurrence.claim))
        for occurrence in occurrences
    ]
