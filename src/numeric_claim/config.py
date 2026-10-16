"""Reading and validating the configuration file."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numeric_claim.errors import ConfigError
from numeric_claim.globs import bind_files
from numeric_claim.relations import QUANTITIES, RELATIONS

DEFAULT_LOCK = "claims.lock.json"
DEFAULT_DISTRIBUTIONS = ("numpy", "scipy", "mpmath")
DEFAULT_SCAN = ("**/*.md", "**/*.tex")
DEFAULT_ALPHA = 0.01
_TOP_LEVEL_KEYS = frozenset({"claims", "lock", "distributions", "scan", "exclude", "alpha", "bind"})
_BIND_KEYS = frozenset({"files", "pattern", "claim", "relation", "quantity"})


@dataclass(frozen=True)
class Binding:
    """A ``[[bind]]`` entry: a pattern applied to files and tied to claims."""

    files: tuple[str, ...]
    pattern: re.Pattern[str]
    claims: tuple[str, ...]
    relation: str
    quantity: str


@dataclass(frozen=True)
class Config:
    """A validated configuration with every path made absolute."""

    path: Path
    directory: Path
    root: Path
    claims: tuple[Path, ...]
    lock: Path
    distributions: tuple[str, ...]
    scan: tuple[str, ...]
    exclude: tuple[str, ...]
    alpha: float
    bindings: tuple[Binding, ...]


def _string_list(
    table: dict[str, Any],
    key: str,
    where: str,
    *,
    required: bool = False,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if key not in table:
        if required:
            raise ConfigError(f"{where}: missing required key {key!r}")
        return default
    value = table[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{where}: {key!r} must be a list of strings")
    if required and not value:
        raise ConfigError(f"{where}: {key!r} must not be empty")
    return tuple(value)


def _alpha(table: dict[str, Any], where: str) -> float:
    value = table.get("alpha", DEFAULT_ALPHA)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise ConfigError(f"{where}: 'alpha' must be a number between 0 and 1, got {value!r}")
    return float(value)


def _check_keys(table: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")


def _binding(index: int, table: dict[str, Any], root: Path) -> Binding:
    where = f"[[bind]] entry {index}"
    _check_keys(table, _BIND_KEYS, where)
    globs = _string_list(table, "files", where, required=True)
    text = table.get("pattern")
    if not isinstance(text, str):
        raise ConfigError(f"{where}: 'pattern' is required and must be a string")
    try:
        pattern = re.compile(text, re.MULTILINE)
    except re.error as exc:
        raise ConfigError(f"{where}: invalid pattern {text!r}: {exc}") from exc
    if pattern.groups != 1:
        raise ConfigError(
            f"{where}: pattern {text!r} must have exactly one capture group, found {pattern.groups}"
        )
    claim = table.get("claim")
    claims = (
        (claim,) if isinstance(claim, str) else _string_list(table, "claim", where, required=True)
    )
    relation = table.get("relation", "eq")
    if relation not in RELATIONS:
        raise ConfigError(
            f"{where}: unknown relation {relation!r}; use one of {', '.join(RELATIONS)}"
        )
    quantity = table.get("quantity", "value")
    if quantity not in QUANTITIES:
        raise ConfigError(
            f"{where}: unknown quantity {quantity!r}; use one of {', '.join(QUANTITIES)}"
        )
    files: list[str] = []
    for glob in globs:
        matched = bind_files(root, glob)
        if not matched:
            raise ConfigError(f"{where}: {glob!r} matches no file under {root}")
        files.extend(name for name in matched if name not in files)
    return Binding(tuple(files), pattern, claims, relation, quantity)


def load_config(path: Path, *, root: Path | None = None, lock: Path | None = None) -> Config:
    """Load and validate a configuration file; ``root`` and ``lock`` override its defaults."""
    path = path.resolve()
    try:
        with path.open("rb") as handle:
            table = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    where = path.name
    _check_keys(table, _TOP_LEVEL_KEYS, where)
    directory = path.parent
    modules = _string_list(table, "claims", where, required=True)
    lock_name = table.get("lock", DEFAULT_LOCK)
    if not isinstance(lock_name, str):
        raise ConfigError(f"{where}: 'lock' must be a string")
    base = (root or directory).resolve()
    if not base.is_dir():
        raise ConfigError(f"root {base} is not a directory")
    bind_tables = table.get("bind", [])
    if not isinstance(bind_tables, list) or not all(isinstance(t, dict) for t in bind_tables):
        raise ConfigError(f"{where}: 'bind' must be an array of tables ([[bind]])")
    return Config(
        path=path,
        directory=directory,
        root=base,
        claims=tuple(directory / name for name in modules),
        lock=lock.resolve() if lock is not None else directory / lock_name,
        distributions=_string_list(table, "distributions", where, default=DEFAULT_DISTRIBUTIONS),
        scan=_string_list(table, "scan", where, default=DEFAULT_SCAN),
        exclude=_string_list(table, "exclude", where),
        alpha=_alpha(table, where),
        bindings=tuple(_binding(i, t, base) for i, t in enumerate(bind_tables, start=1)),
    )
