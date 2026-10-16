"""Monte Carlo claims: replicate seeds, summaries, Student's t tests and Holm's procedure."""

import hashlib
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from numeric_claim.literals import Literal

MAX_TERMS = 10_000
EPSILON = 1e-15
TINY = 1e-300


@dataclass(frozen=True)
class Summary:
    """The mean and standard deviation of a random claim's replicate values, and their number."""

    mean: float
    sd: float
    replicates: int

    @property
    def se(self) -> float:
        """The standard error of the mean."""
        return self.sd / math.sqrt(self.replicates)


@dataclass(frozen=True)
class TTest:
    """A printed number tested against a random claim: t, its p-value before and after Holm."""

    t: float
    p: float
    p_holm: float
    label: str


def seeds(name: str, replicates: int) -> list[int]:
    """The seed passed to each replicate of the claim ``name``."""
    return [
        int.from_bytes(hashlib.sha256(f"{name}/{index}".encode()).digest()[:8], "big")
        for index in range(replicates)
    ]


def summarize(values: Sequence[float]) -> Summary:
    """The mean, standard deviation and number of a random claim's replicate values."""
    return Summary(statistics.fmean(values), statistics.stdev(values), len(values))


def _guarded(value: float) -> float:
    return value if abs(value) >= TINY else TINY


def _fraction(x: float, a: float, b: float) -> float:
    """The continued fraction of ``I_x(a, b)``, evaluated by the modified Lentz method."""
    c, d = 1.0, 1.0 / _guarded(1.0 - (a + b) * x / (a + 1.0))
    h = d
    for m in range(1, MAX_TERMS + 1):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 / _guarded(1.0 + numerator * d)
            c = _guarded(1.0 + numerator / c)
            h *= d * c
        if abs(d * c - 1.0) < EPSILON:
            return h
    raise ArithmeticError(f"the incomplete beta function did not converge for x={x!r}")


def _incomplete_beta(x: float, y: float, a: float, b: float) -> float:
    """The regularized incomplete beta function ``I_x(a, b)``, given ``y = 1 - x``."""
    if x == 0 or y == 0:
        return 0.0 if x == 0 else 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
    )
    if x > (a + 1) / (a + b + 2):
        return 1.0 - front * _fraction(y, b, a) / b
    return front * _fraction(x, a, b) / a


def t_sf(t: float, df: float) -> float:
    """The upper tail ``P(T >= t)`` of Student's t distribution with ``df`` degrees of freedom."""
    if t < 0:
        return 1.0 - t_sf(-t, df)
    square = t * t
    if math.isinf(square):
        return 0.0
    return 0.5 * _incomplete_beta(df / (df + square), square / (df + square), df / 2, 0.5)


def t_test(
    relation: str, literal: Literal, summary: Summary, tolerance: float
) -> tuple[float, float]:
    """t and the p-value of a printed number read as a statement about the expected value.

    t is the distance from the mean to the nearest value at which the statement holds, in standard
    errors; ``eq`` is tested two-sided and the bounds one-sided.
    """
    printed, mean = Fraction(literal.value), Fraction(summary.mean)
    if relation == "eq":
        allowance = Fraction(0 if literal.is_integer else literal.half_unit) + Fraction(tolerance)
        distance = max(Fraction(0), printed - allowance - mean, mean - printed - allowance)
    elif relation in ("lt", "le"):
        distance = max(Fraction(0), mean - printed)
    elif relation in ("gt", "ge"):
        distance = max(Fraction(0), printed - mean)
    else:
        raise ValueError(f"unknown relation {relation!r}")
    if distance == 0:
        return 0.0, 1.0
    if summary.se == 0:
        return math.inf, 0.0
    t = float(distance) / summary.se
    tail = t_sf(t, summary.replicates - 1)
    return t, min(1.0, 2 * tail) if relation == "eq" else tail


def welch(first: Summary, second: Summary) -> float:
    """The two-sided p-value of Welch's test that two summaries share their mean."""
    v1 = first.sd * first.sd / first.replicates
    v2 = second.sd * second.sd / second.replicates
    if v1 + v2 == 0:
        return 0.0 if first.mean != second.mean else 1.0
    t = (first.mean - second.mean) / math.sqrt(v1 + v2)
    u1, u2 = v1 / (v1 + v2), v2 / (v1 + v2)
    df = 1 / (u1 * u1 / (first.replicates - 1) + u2 * u2 / (second.replicates - 1))
    return min(1.0, 2 * t_sf(abs(t), df))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment of ``p_values``, returned in their order; ties keep it."""
    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(sorted(range(count), key=p_values.__getitem__)):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
