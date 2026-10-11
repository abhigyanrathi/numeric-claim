"""Claim declaration: the decorator, its metadata and the context passed to claim functions."""

import inspect
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeGuard

from numeric_claim.literals import Literal, parse_literal

type Value = int | float

DEFAULT_RTOL = 1e-12
DEFAULT_DPS = 50
MIN_DPS = 20
MAX_JITTER = 1e-6
ATTRIBUTE = "__numeric_claim__"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class ClaimContext:
    """Where the project and its configuration live."""

    root: Path
    config_dir: Path


@dataclass(frozen=True)
class ClosedForm:
    """A reference computed with mpmath at ``dps`` significant digits and treated as exact."""

    function: Callable[[Any], Any]
    dps: int
    text: ClassVar[str] = "closed form"


@dataclass(frozen=True)
class Probes:
    """What an error budget varies: keyword-only parameters, each mapping in signature order."""

    defaults: dict[str, Value]
    nudged: tuple[str, ...]
    jitter: dict[str, float]
    ladders: dict[str, tuple[Value, ...]]


@dataclass(frozen=True)
class ClaimSpec:
    """What a claim declares about itself."""

    name: str
    kind: str
    rtol: float
    atol: float
    packages: tuple[str, ...]
    paths: tuple[str, ...]
    reference: Literal | ClosedForm | None = None
    probes: Probes | None = None

    def tolerance(self, value: Value) -> float:
        """Largest difference from ``value`` that still counts as agreement."""
        if self.kind == "exact":
            return 0.0
        return max(self.atol, self.rtol * abs(value))


def closed_form(fn: Callable[[Any], Any], *, dps: int = DEFAULT_DPS) -> ClosedForm:
    """A reference given in closed form: ``fn`` takes the mpmath module and returns a number."""
    if not isinstance(dps, int) or dps < MIN_DPS:
        raise ValueError(f"dps must be an int of at least {MIN_DPS}, got {dps!r}")
    return ClosedForm(fn, dps)


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


def _reference(reference: str | ClosedForm | None) -> Literal | ClosedForm | None:
    if reference is None or isinstance(reference, ClosedForm):
        return reference
    if not isinstance(reference, str):
        raise ValueError(
            f"reference must be a number written as a string or closed_form(...), got {reference!r}"
        )
    try:
        return parse_literal(reference)
    except ValueError as exc:
        raise ValueError(f"invalid reference: {exc}") from exc


def _mapping[T](label: str, value: Mapping[str, T] | None) -> Mapping[str, T]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must map parameter names to their settings, got {value!r}")
    return value


def _numeric(value: object) -> TypeGuard[Value]:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))


def _ladder(name: str, default: Value, settings: Iterable[Value]) -> tuple[Value, ...]:
    whole = isinstance(default, int)
    expected = "an int" if whole else "an int or a finite float"
    try:
        rungs = tuple(settings)
    except TypeError as exc:
        raise ValueError(f"the ladder for {name!r} must be a sequence of settings") from exc
    if not rungs:
        raise ValueError(f"the ladder for {name!r} is empty")
    for setting in rungs:
        if not _numeric(setting) or (whole and not isinstance(setting, int)):
            raise ValueError(f"a ladder setting for {name!r} must be {expected}, got {setting!r}")
    return tuple(setting if isinstance(setting, int) else float(setting) for setting in rungs)


def _amplitude(name: str, default: Value, amplitude: object) -> float:
    if not isinstance(default, float):
        raise ValueError(f"jitter needs a float default, and that of {name!r} is {default!r}")
    if not _numeric(amplitude) or not 0 < amplitude <= MAX_JITTER:
        raise ValueError(
            f"the jitter amplitude for {name!r} must be in (0, {MAX_JITTER}], got {amplitude!r}"
        )
    return float(amplitude)


def _probes(
    function: Callable[..., Any],
    kind: str,
    ladders: Mapping[str, Iterable[Value]],
    jitter: Mapping[str, float],
) -> Probes | None:
    defaults: dict[str, Value] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind != parameter.KEYWORD_ONLY:
            continue
        if kind == "exact":
            raise ValueError("an exact claim takes no keyword-only parameters")
        if parameter.default is parameter.empty:
            raise ValueError(f"keyword-only parameter {parameter.name!r} needs a default")
        if not _numeric(parameter.default):
            raise ValueError(
                f"the default of {parameter.name!r} must be an int or a finite float, "
                f"got {parameter.default!r}"
            )
        defaults[parameter.name] = parameter.default
    for label, declared in (("ladders", ladders), ("jitter", jitter)):
        for name in declared:
            if name not in defaults:
                raise ValueError(f"{label} names {name!r}, which is not a keyword-only parameter")
    if not defaults:
        return None
    varied = {*ladders, *jitter}
    nudged = tuple(
        name
        for name, default in defaults.items()
        if isinstance(default, float) and default != 0 and name not in varied
    )
    amplitudes = {
        name: _amplitude(name, default, jitter[name])
        for name, default in defaults.items()
        if name in jitter
    }
    rungs = {
        name: _ladder(name, default, ladders[name])
        for name, default in defaults.items()
        if name in ladders
    }
    if not nudged and not amplitudes:
        raise ValueError("an error budget needs a nonzero float input to nudge or a jitter")
    return Probes(defaults, nudged, amplitudes, rungs)


def claim[F: Callable[..., Any]](
    name: str,
    *,
    kind: str = "exact",
    rtol: float | None = None,
    atol: float | None = None,
    packages: Iterable[str] = (),
    paths: Iterable[str] = (),
    reference: str | ClosedForm | None = None,
    ladders: Mapping[str, Iterable[Value]] | None = None,
    jitter: Mapping[str, float] | None = None,
) -> Callable[[F], F]:
    """Declare a computation whose value documents may quote.

    A seeded claim may name the ``reference`` it reproduces. Keyword-only parameters with numeric
    defaults give it an error budget: ``ladders`` lists refinements of some of them, ``jitter``
    relative amplitudes for others, and the remaining nonzero float defaults are nudged.
    """
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid claim name {name!r}: use letters, digits, '_', '.' and '-'")
    if kind == "exact":
        if rtol is not None or atol is not None:
            raise ValueError("an exact claim takes no rtol or atol")
        if reference is not None or ladders is not None or jitter is not None:
            raise ValueError("an exact claim takes no reference, ladders or jitter")
        tolerances = (0.0, 0.0)
    elif kind == "seeded":
        tolerances = (_tolerance("rtol", rtol, DEFAULT_RTOL), _tolerance("atol", atol, 0.0))
    else:
        raise ValueError(f"unknown claim kind {kind!r}: use 'exact' or 'seeded'")
    sources = (_strings("packages", packages), _strings("paths", paths))
    target = _reference(reference)
    settings = (_mapping("ladders", ladders), _mapping("jitter", jitter))

    def decorate(function: F) -> F:
        probes = _probes(function, kind, *settings)
        setattr(function, ATTRIBUTE, ClaimSpec(name, kind, *tolerances, *sources, target, probes))
        return function

    return decorate


def spec_of(candidate: object) -> ClaimSpec | None:
    """The claim metadata attached to ``candidate``, if any."""
    spec = getattr(candidate, ATTRIBUTE, None)
    return spec if isinstance(spec, ClaimSpec) else None
