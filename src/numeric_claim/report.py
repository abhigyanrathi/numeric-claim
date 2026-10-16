"""Rendering command results as text for the terminal and as data for ``--json``."""

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from numeric_claim.claim import ClaimSpec, Value
from numeric_claim.engine import ClaimOutcome, LockOutcome, OccurrenceResult
from numeric_claim.loader import LoadedClaim
from numeric_claim.scan import Occurrence


@dataclass(frozen=True)
class Report:
    """A command's result: terminal text, JSON-ready data and the exit status."""

    text: str
    data: dict[str, Any]
    exit_code: int


def _show(value: Value | None) -> str:
    return "-" if value is None else repr(value)


def _float(number: Decimal | float | None) -> float | None:
    """``number`` as a float; None if there is none or it is beyond a float's range."""
    if number is None:
        return None
    converted = float(number)
    return converted if math.isfinite(converted) else None


def _brief(number: float | None) -> str:
    return "-" if number is None else f"{number:.3g}"


def _tally(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    return ", ".join(f"{counts[label]} {label}" for label in sorted(counts)) or "none"


def _occurrence_data(occurrence: Occurrence) -> dict[str, Any]:
    return {
        "file": occurrence.file,
        "line": occurrence.line,
        "claim": occurrence.claim,
        "relation": occurrence.relation,
        "literal": occurrence.literal,
        "origin": occurrence.origin,
        "quantity": occurrence.quantity,
    }


def _budget_data(outcome: ClaimOutcome) -> dict[str, Any] | None:
    budget, found = outcome.budget, outcome.attribution
    if budget is None:
        return None
    return {
        "nudge": budget.nudge,
        "jitter": budget.jitter,
        "sigma": budget.sigma,
        "ladders": {
            name: [
                {
                    "setting": rung.setting,
                    "value": rung.value,
                    "error": None if found is None else _float(found.ladders[name][index]),
                }
                for index, rung in enumerate(rungs)
            ]
            for name, rungs in budget.ladders.items()
        },
        "gains": None if found is None else {n: _float(g) for n, g in found.gains.items()},
        "error": None if found is None else _float(found.error),
        "attribution": None if found is None else found.label,
    }


def _random_data(outcome: ClaimOutcome) -> dict[str, Any] | None:
    summary = outcome.summary
    if summary is None:
        return None
    return {
        "replicates": summary.replicates,
        "mean": summary.mean,
        "sd": summary.sd,
        "se": summary.se,
    }


def _claim_line(outcome: ClaimOutcome) -> str:
    line = f"claim {outcome.name} [{outcome.kind}] {outcome.status} value={_show(outcome.value)}"
    if outcome.source != "-":
        line += f" ({outcome.source})"
    budget, found, summary = outcome.budget, outcome.attribution, outcome.summary
    if budget is not None:
        line += f" sigma={_brief(budget.sigma)}"
        line += f" (nudge={_brief(budget.nudge)}, jitter={_brief(budget.jitter)})"
    if found is not None:
        line += f" error={_brief(_float(found.error))} attribution={found.label}"
    if summary is not None:
        line += f" mean={_brief(summary.mean)} sd={_brief(summary.sd)} se={_brief(summary.se)}"
        line += f" R={summary.replicates}"
    if outcome.note:
        line += f": {outcome.note}"
    return line


def _occurrence_line(result: OccurrenceResult) -> str:
    occurrence, found, test = result.occurrence, result.verdict, result.test
    line = (
        f"{occurrence.file}:{occurrence.line} {occurrence.claim} {occurrence.relation} "
        f"{occurrence.literal} value={_show(result.value)} {result.result}"
    )
    if occurrence.quantity == "error" and result.compared is not None:
        line += f" quantity=error compared={_brief(_float(result.compared))}"
    elif found is not None:
        line += f" quantity=value compared={_show(result.value)}"
    if found is not None:
        line += f" margin={_brief(_float(found.margin))} 3sigma={_brief(_float(found.band))}"
        line += f" verdict={found.label}"
    if test is not None:
        line += f" t={_brief(test.t)} p={_brief(test.p)} p_holm={_brief(test.p_holm)}"
        line += f" verdict={test.label}"
    if result.note:
        line += f": {result.note}"
    return line


def _verdict(result: OccurrenceResult) -> str | None:
    if result.verdict is not None:
        return result.verdict.label
    return None if result.test is None else result.test.label


def check_report(
    outcomes: Sequence[ClaimOutcome], results: Sequence[OccurrenceResult], alpha: float
) -> Report:
    """Report for ``check``: exit 1 unless every claim is ok and every occurrence passes."""
    failing = any(o.status != "ok" for o in outcomes) or any(r.result != "pass" for r in results)
    exit_code = 1 if failing else 0
    lines = [_claim_line(outcome) for outcome in outcomes]
    lines += [_occurrence_line(result) for result in results]
    lines.append(
        f"{len(outcomes)} claims ({_tally(o.status for o in outcomes)}), "
        f"{len(results)} occurrences ({_tally(r.result for r in results)}): exit {exit_code}"
    )
    data = {
        "command": "check",
        "exit_code": exit_code,
        "alpha": alpha,
        "claims": [
            {
                "name": o.name,
                "kind": o.kind,
                "status": o.status,
                "value": o.value,
                "locked_value": o.locked,
                "source": o.source,
                "note": o.note,
                "reference": o.reference,
                "budget": _budget_data(o),
                "random": _random_data(o),
            }
            for o in outcomes
        ],
        "occurrences": [
            {
                **_occurrence_data(r.occurrence),
                "value": r.value,
                "result": r.result,
                "note": r.note,
                "compared": _float(r.compared),
                "margin": None if r.verdict is None else _float(r.verdict.margin),
                "band": None if r.verdict is None else _float(r.verdict.band),
                "verdict": _verdict(r),
                "t": None if r.test is None else _float(r.test.t),
                "p": None if r.test is None else r.test.p,
                "p_holm": None if r.test is None else r.test.p_holm,
            }
            for r in results
        ],
    }
    return Report("\n".join(lines) + "\n", data, exit_code)


def lock_report(outcomes: Sequence[LockOutcome], path: Path, *, written: bool) -> Report:
    """Report for ``lock``: exit 1, and nothing written, if any claim failed."""
    lines = []
    for outcome in outcomes:
        if outcome.entry is not None:
            lines.append(f"locked {outcome.name} = {_show(outcome.entry.value)}")
        else:
            lines.append(f"error {outcome.name}: {outcome.error}")
    if written:
        lines.append(f"wrote {len(outcomes)} claims to {path}")
    else:
        failed = sum(outcome.entry is None for outcome in outcomes)
        lines.append(f"nothing written: {failed} of {len(outcomes)} claims failed")
    exit_code = 0 if written else 1
    data = {
        "command": "lock",
        "exit_code": exit_code,
        "lock": str(path),
        "written": written,
        "claims": [
            {
                "name": o.name,
                "value": o.entry.value if o.entry is not None else None,
                "error": o.error,
            }
            for o in outcomes
        ],
    }
    return Report("\n".join(lines) + "\n", data, exit_code)


def _declared(spec: ClaimSpec) -> dict[str, Any]:
    probes = spec.probes
    return {
        "reference": None if spec.reference is None else spec.reference.text,
        "nudged": [] if probes is None else list(probes.nudged),
        "ladders": {} if probes is None else {n: list(s) for n, s in probes.ladders.items()},
        "jitter": {} if probes is None else dict(probes.jitter),
    }


def _declared_text(declared: dict[str, Any]) -> str:
    ladders = ";".join(
        f"{name}:{','.join(map(repr, settings))}" for name, settings in declared["ladders"].items()
    )
    jitter = ",".join(f"{name}:{amplitude!r}" for name, amplitude in declared["jitter"].items())
    return (
        f"reference={declared['reference'] or '-'} nudged={','.join(declared['nudged']) or '-'} "
        f"ladders={ladders or '-'} jitter={jitter or '-'}"
    )


def _replicates(spec: ClaimSpec) -> dict[str, int]:
    return {} if spec.replicates is None else {"replicates": spec.replicates}


def list_report(claims: Sequence[LoadedClaim], occurrences: Sequence[Occurrence]) -> Report:
    """Report for ``list``: declared claims and found occurrences, nothing computed.

    A random claim also shows its number of replicates.
    """
    lines = []
    for loaded in claims:
        spec = loaded.spec
        packages = ",".join(spec.packages) or "-"
        paths = ",".join(spec.paths) or "-"
        replicates = "" if spec.replicates is None else f" replicates={spec.replicates}"
        lines.append(
            f"claim {spec.name} [{spec.kind}] packages={packages} paths={paths} "
            f"{_declared_text(_declared(spec))}{replicates}"
        )
    for occurrence in occurrences:
        lines.append(
            f"{occurrence.file}:{occurrence.line} {occurrence.claim} "
            f"{occurrence.relation} {occurrence.literal} quantity={occurrence.quantity}"
        )
    lines.append(f"{len(claims)} claims, {len(occurrences)} occurrences")
    data = {
        "command": "list",
        "exit_code": 0,
        "claims": [
            {
                "name": c.spec.name,
                "kind": c.spec.kind,
                "packages": list(c.spec.packages),
                "paths": list(c.spec.paths),
                **_declared(c.spec),
                **_replicates(c.spec),
            }
            for c in claims
        ],
        "occurrences": [_occurrence_data(o) for o in occurrences],
    }
    return Report("\n".join(lines) + "\n", data, 0)
