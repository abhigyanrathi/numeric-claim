"""Rendering command results as text for the terminal and as data for ``--json``."""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numeric_claim.claim import Value
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
    }


def _claim_line(outcome: ClaimOutcome) -> str:
    line = f"claim {outcome.name} [{outcome.kind}] {outcome.status} value={_show(outcome.value)}"
    if outcome.source != "-":
        line += f" ({outcome.source})"
    if outcome.note:
        line += f": {outcome.note}"
    return line


def _occurrence_line(result: OccurrenceResult) -> str:
    occurrence = result.occurrence
    line = (
        f"{occurrence.file}:{occurrence.line} {occurrence.claim} {occurrence.relation} "
        f"{occurrence.literal} value={_show(result.value)} {result.result}"
    )
    if result.note:
        line += f": {result.note}"
    return line


def check_report(outcomes: Sequence[ClaimOutcome], results: Sequence[OccurrenceResult]) -> Report:
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
        "claims": [
            {
                "name": o.name,
                "kind": o.kind,
                "status": o.status,
                "value": o.value,
                "locked_value": o.locked,
                "source": o.source,
                "note": o.note,
            }
            for o in outcomes
        ],
        "occurrences": [
            {
                **_occurrence_data(r.occurrence),
                "value": r.value,
                "result": r.result,
                "note": r.note,
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


def list_report(claims: Sequence[LoadedClaim], occurrences: Sequence[Occurrence]) -> Report:
    """Report for ``list``: declared claims and found occurrences, nothing computed."""
    lines = []
    for loaded in claims:
        spec = loaded.spec
        packages = ",".join(spec.packages) or "-"
        paths = ",".join(spec.paths) or "-"
        lines.append(f"claim {spec.name} [{spec.kind}] packages={packages} paths={paths}")
    for occurrence in occurrences:
        lines.append(
            f"{occurrence.file}:{occurrence.line} {occurrence.claim} "
            f"{occurrence.relation} {occurrence.literal}"
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
            }
            for c in claims
        ],
        "occurrences": [_occurrence_data(o) for o in occurrences],
    }
    return Report("\n".join(lines) + "\n", data, 0)
