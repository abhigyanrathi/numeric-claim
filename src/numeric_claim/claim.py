"""Claim declaration: the decorator, its metadata and the context passed to claim functions."""

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

type Value = int | float

DEFAULT_RTOL = 1e-12
ATTRIBUTE = "__numeric_claim__"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class ClaimContext:
    """Where the project and its configuration live."""

    root: Path
    config_dir: Path


@dataclass(frozen=True)
class ClaimSpec:
    """What a claim declares about itself."""

    name: str
    kind: str
    rtol: float
    atol: float
    packages: tuple[str, ...]
    paths: tuple[str, ...]

    def tolerance(self, value: Value) -> float:
        """Largest difference from ``value`` that still counts as agreement."""
        if self.kind == "exact":
            return 0.0
        return max(self.atol, self.rtol * abs(value))


def _tolerance(label: str, value: float | None, default: float) -> float:
    if value is None:
        return default
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite number >= 0, got {value!r}")
    return float(value)


def _strings(label: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{label} must be a sequence of strings, not a single string")
    return tuple(values)


def claim[F: Callable[..., Any]](
    name: str,
    *,
    kind: str = "exact",
    rtol: float | None = None,
    atol: float | None = None,
    packages: Iterable[str] = (),
    paths: Iterable[str] = (),
) -> Callable[[F], F]:
    """Declare a computation whose value documents may quote."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid claim name {name!r}: use letters, digits, '_', '.' and '-'")
    if kind == "exact":
        if rtol is not None or atol is not None:
            raise ValueError("an exact claim takes no rtol or atol")
        tolerances = (0.0, 0.0)
    elif kind == "seeded":
        tolerances = (_tolerance("rtol", rtol, DEFAULT_RTOL), _tolerance("atol", atol, 0.0))
    else:
        raise ValueError(f"unknown claim kind {kind!r}: use 'exact' or 'seeded'")
    spec = ClaimSpec(
        name, kind, *tolerances, _strings("packages", packages), _strings("paths", paths)
    )

    def decorate(function: F) -> F:
        setattr(function, ATTRIBUTE, spec)
        return function

    return decorate


def spec_of(candidate: object) -> ClaimSpec | None:
    """The claim metadata attached to ``candidate``, if any."""
    spec = getattr(candidate, ATTRIBUTE, None)
    return spec if isinstance(spec, ClaimSpec) else None
