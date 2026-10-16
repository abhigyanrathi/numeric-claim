"""The committed lock file: its entries, reading and deterministic writing."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numeric_claim.budget import Budget, Rung
from numeric_claim.claim import Value
from numeric_claim.errors import ConfigError
from numeric_claim.montecarlo import Summary

FORMAT = 1
KINDS = ("exact", "seeded", "random")


@dataclass(frozen=True)
class LockEntry:
    """One locked claim value with the fingerprint it was computed under.

    A random claim's value is the mean of its replicates, which ``summary`` describes.
    """

    kind: str
    value: Value
    fingerprint: str
    environment: dict[str, Any]
    rtol: float = 0.0
    atol: float = 0.0
    budget: Budget | None = None
    summary: Summary | None = None


def _encode_value(value: Value) -> int | dict[str, str]:
    if isinstance(value, int):
        return value
    return {"hex": value.hex(), "repr": repr(value)}


def _decode_value(raw: Any) -> Value:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("hex"), str):
        return float.fromhex(raw["hex"])
    raise ValueError(f"unreadable value {raw!r}")


def _number(raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"expected a number, got {raw!r}")
    return float(raw)


def _encode_deviation(deviation: float | None) -> int | dict[str, str] | None:
    return None if deviation is None else _encode_value(deviation)


def _decode_deviation(raw: Any) -> float | None:
    return None if raw is None else float(_decode_value(raw))


def _encode_budget(budget: Budget) -> dict[str, Any]:
    return {
        "jitter": _encode_deviation(budget.jitter),
        "ladders": {
            name: [{"setting": rung.setting, "value": _encode_value(rung.value)} for rung in rungs]
            for name, rungs in budget.ladders.items()
        },
        "nudge": _encode_deviation(budget.nudge),
    }


def _decode_rung(raw: Any) -> Rung:
    if not isinstance(raw, dict):
        raise ValueError(f"unreadable ladder entry {raw!r}")
    setting = raw.get("setting")
    if isinstance(setting, bool) or not isinstance(setting, int | float):
        raise ValueError(f"expected a number, got {setting!r}")
    return Rung(setting, float(_decode_value(raw.get("value"))))


def _decode_budget(raw: Any) -> Budget:
    if not isinstance(raw, dict) or not isinstance(raw.get("ladders"), dict):
        raise ValueError("budget is not an object with ladders")
    nudge, jitter = _decode_deviation(raw.get("nudge")), _decode_deviation(raw.get("jitter"))
    if nudge is None and jitter is None:
        raise ValueError("budget has neither nudge nor jitter")
    ladders: dict[str, tuple[Rung, ...]] = {}
    for name, rungs in raw["ladders"].items():
        if not isinstance(rungs, list):
            raise ValueError(f"ladder {name!r} is not a list")
        ladders[name] = tuple(_decode_rung(rung) for rung in rungs)
    return Budget(nudge, jitter, ladders)


def _decode_summary(raw: dict[str, Any], mean: Value) -> Summary:
    replicates = raw.get("replicates")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError(f"expected at least 2 replicates, got {replicates!r}")
    summary = Summary(float(mean), float(_decode_value(raw.get("sd"))), replicates)
    if not math.isfinite(summary.mean) or not math.isfinite(summary.sd) or summary.sd < 0:
        raise ValueError(f"unusable mean or standard deviation: {summary.mean!r}, {summary.sd!r}")
    return summary


def _encode_entry(entry: LockEntry) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "environment": entry.environment,
        "fingerprint": entry.fingerprint,
        "kind": entry.kind,
        "value": _encode_value(entry.value),
    }
    if entry.kind in ("seeded", "random"):
        encoded["rtol"] = entry.rtol
        encoded["atol"] = entry.atol
    if entry.budget is not None:
        encoded["budget"] = _encode_budget(entry.budget)
    if entry.summary is not None:
        encoded["replicates"] = entry.summary.replicates
        encoded["sd"] = _encode_value(entry.summary.sd)
    return encoded


def _decode_entry(raw: Any) -> LockEntry:
    if not isinstance(raw, dict):
        raise ValueError("entry is not an object")
    kind, fingerprint, environment = raw.get("kind"), raw.get("fingerprint"), raw.get("environment")
    if kind not in KINDS or not isinstance(fingerprint, str) or not isinstance(environment, dict):
        raise ValueError("entry lacks a valid kind, fingerprint or environment")
    value = _decode_value(raw.get("value"))
    if kind == "seeded":
        rtol, atol = _number(raw.get("rtol")), _number(raw.get("atol"))
        budget = None if raw.get("budget") is None else _decode_budget(raw["budget"])
        return LockEntry(kind, value, fingerprint, environment, rtol, atol, budget)
    if kind == "random":
        rtol, atol = _number(raw.get("rtol")), _number(raw.get("atol"))
        summary = _decode_summary(raw, value)
        return LockEntry(kind, summary.mean, fingerprint, environment, rtol, atol, None, summary)
    return LockEntry(kind, value, fingerprint, environment)


def dumps(entries: Mapping[str, LockEntry]) -> str:
    """The lock file text for ``entries``: sorted keys, two-space indent, final newline."""
    claims = {name: _encode_entry(entry) for name, entry in entries.items()}
    document = {"claims": claims, "format": FORMAT}
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write(path: Path, entries: Mapping[str, LockEntry]) -> None:
    """Write the lock file with ``\\n`` line endings on every platform."""
    try:
        path.write_text(dumps(entries), encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ConfigError(f"cannot write lock file {path}: {exc.strerror or exc}") from exc


def loads(text: str, where: str) -> dict[str, LockEntry]:
    """The entries of lock file text; ``where`` names the file in errors."""
    try:
        document = json.loads(text)
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise ValueError(f"not a format {FORMAT} lock file")
        claims = document.get("claims")
        if not isinstance(claims, dict):
            raise ValueError("'claims' is not an object")
        return {name: _decode_entry(raw) for name, raw in claims.items()}
    except (ValueError, OverflowError) as exc:
        raise ConfigError(f"invalid lock file {where}: {exc}") from exc


def read(path: Path) -> dict[str, LockEntry]:
    """Read a lock file; a missing file is an empty lock."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read lock file {path}: {exc}") from exc
    return loads(text, str(path))
