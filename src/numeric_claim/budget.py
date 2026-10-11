"""Error budgets: the probe draws, what the probes measure, verdicts and attribution."""

import decimal
import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from numeric_claim.claim import Probes, Value
from numeric_claim.literals import Literal
from numeric_claim.relations import holds

DRAWS = 32
MAX_STEPS = 8
BAND = 3
EXACT = decimal.Context(prec=decimal.MAX_PREC, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)


@dataclass(frozen=True)
class Rung:
    """One ladder setting and the value the claim gave at it."""

    setting: Value
    value: float


@dataclass(frozen=True)
class Budget:
    """What the probes measured: two standard deviations and the value at each ladder setting."""

    nudge: float | None
    jitter: float | None
    ladders: dict[str, tuple[Rung, ...]]

    @property
    def sigma(self) -> float:
        """The rounding noise: the larger of the standard deviations that were measured."""
        return max(deviation for deviation in (self.nudge, self.jitter) if deviation is not None)

    def arranged(self, probes: Probes) -> "Budget | None":
        """This budget with its ladders in signature order; None if ``probes`` measure otherwise."""
        settings = {
            name: tuple(rung.setting for rung in rungs) for name, rungs in self.ladders.items()
        }
        if (
            settings != probes.ladders
            or (self.nudge is not None) != bool(probes.nudged)
            or (self.jitter is not None) != bool(probes.jitter)
        ):
            return None
        ladders = {name: self.ladders[name] for name in probes.ladders}
        return Budget(self.nudge, self.jitter, ladders)


@dataclass(frozen=True)
class Verdict:
    """How a printed number stands against a budgeted claim; ``band`` is 3 sigma."""

    label: str
    margin: Decimal
    band: Decimal


@dataclass(frozen=True)
class Attribution:
    """A budgeted claim's error against its reference, on its ladders too, and what dominates it."""

    error: Decimal
    ladders: dict[str, tuple[Decimal, ...]]
    gains: dict[str, Decimal]
    label: str


def _nudged(default: float, rng: random.Random) -> float:
    steps = 1 + int(MAX_STEPS * rng.random())
    toward = math.inf if rng.random() < 0.5 else -math.inf
    return math.nextafter(default, toward, steps=steps)


def _jittered(default: float, amplitude: float, rng: random.Random) -> float:
    u = 2.0 * rng.random() - 1.0
    return default * (1.0 + amplitude * u)


def draws(name: str, probes: Probes) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """The keyword arguments of the nudged calls and of the jittered calls, seeded by ``name``."""
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    nudged = [
        {parameter: _nudged(float(probes.defaults[parameter]), rng) for parameter in probes.nudged}
        for _ in range(DRAWS if probes.nudged else 0)
    ]
    jittered = [
        {
            parameter: _jittered(float(probes.defaults[parameter]), amplitude, rng)
            for parameter, amplitude in probes.jitter.items()
        }
        for _ in range(DRAWS if probes.jitter else 0)
    ]
    return nudged, jittered


def distance(value: Value, reference: Decimal) -> Decimal:
    """``|value - reference|``, exactly."""
    with decimal.localcontext(EXACT):
        return abs(Decimal(value) - reference)


def verdict(
    relation: str,
    compared: Decimal,
    literal: Literal,
    tolerance: float,
    sigma: float,
    half_unit: Decimal | None = None,
) -> Verdict:
    """Judge ``compared`` against a printed number, given the noise ``sigma`` of its claim.

    ``half_unit`` is that of the claim's reference when ``compared`` is the claim's error, and
    None when it is the claim's value.
    """
    printed = literal.value
    with decimal.localcontext(EXACT):
        if relation in ("lt", "le"):
            margin = printed - compared
        elif relation in ("gt", "ge"):
            margin = compared - printed
        else:
            limit = Decimal(0) if literal.is_integer else literal.half_unit + Decimal(tolerance)
            margin = limit - abs(compared - printed)
        band = BAND * Decimal(sigma)
    if half_unit is not None and relation in ("lt", "le") and printed <= half_unit:
        label = "unsupported"
    elif not holds(relation, compared, literal, tolerance):
        label = "violated"
    elif margin <= band:
        label = "rounding-limited"
    else:
        label = "supported"
    return Verdict(label, margin, band)


def dominant(error: Decimal, gains: Mapping[str, Decimal], sigma: float, half_unit: Decimal) -> str:
    """What dominates ``error``: a ladder parameter, rounding, the reference, or nothing found.

    ``gains`` maps each ladder parameter, in signature order, to how far its ladder reduces the
    error; ``half_unit`` is that of the reference.
    """
    best = max(gains, key=gains.__getitem__, default=None)
    with decimal.localcontext(EXACT):
        band = BAND * Decimal(sigma)
        if best is not None and gains[best] > band and 2 * gains[best] >= error:
            return best
        if 2 * band >= error:
            return "rounding"
    return "reference" if error <= half_unit else "unexplained"


def attribute(value: Value, budget: Budget, reference: Literal) -> Attribution:
    """Set a claim's value and its ladder values against its reference."""
    error = distance(value, reference.value)
    ladders = {
        name: tuple(distance(rung.value, reference.value) for rung in rungs)
        for name, rungs in budget.ladders.items()
    }
    with decimal.localcontext(EXACT):
        gains = {name: error - min(errors) for name, errors in ladders.items()}
    label = dominant(error, gains, budget.sigma, reference.half_unit)
    return Attribution(error, ladders, gains, label)
