"""The committed lock file: its entries, reading and deterministic writing."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numeric_claim.claim import Value
from numeric_claim.errors import ConfigError

FORMAT = 1
KINDS = ("exact", "seeded")


@dataclass(frozen=True)
class LockEntry:
    """One locked claim value with the fingerprint it was computed under."""

    kind: str
    value: Value
    fingerprint: str
    environment: dict[str, Any]
    rtol: float = 0.0
    atol: float = 0.0


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


def _encode_entry(entry: LockEntry) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "environment": entry.environment,
        "fingerprint": entry.fingerprint,
        "kind": entry.kind,
        "value": _encode_value(entry.value),
    }
    if entry.kind == "seeded":
        encoded["rtol"] = entry.rtol
        encoded["atol"] = entry.atol
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
        return LockEntry(kind, value, fingerprint, environment, rtol, atol)
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


def read(path: Path) -> dict[str, LockEntry]:
    """Read a lock file; a missing file is an empty lock."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read lock file {path}: {exc}") from exc
    try:
        document = json.loads(text)
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise ValueError(f"not a format {FORMAT} lock file")
        claims = document.get("claims")
        if not isinstance(claims, dict):
            raise ValueError("'claims' is not an object")
        return {name: _decode_entry(raw) for name, raw in claims.items()}
    except (ValueError, OverflowError) as exc:
        raise ConfigError(f"invalid lock file {path}: {exc}") from exc
