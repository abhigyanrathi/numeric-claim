"""Parsing the numbers a document prints."""

import re
from dataclasses import dataclass
from decimal import Decimal

_LITERAL = re.compile(
    r"(?P<sign>[+\-\N{MINUS SIGN}])?"
    r"(?P<integer>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<fraction>\.\d+)?"
    r"(?P<exponent>[eE][+\-\N{MINUS SIGN}]?\d+)?"
)
_MAX_ADJUSTED_EXPONENT = 10_000


@dataclass(frozen=True)
class Literal:
    """A number as printed: its exact decimal value and half a unit of its last digit."""

    text: str
    value: Decimal
    half_unit: Decimal
    is_integer: bool


def parse_literal(text: str) -> Literal:
    """Parse a printed number; raise ValueError if ``text`` is not one."""
    stripped = text.strip()
    match = _LITERAL.fullmatch(stripped)
    if match is None:
        raise ValueError(f"not a number: {text!r}")
    fraction = match["fraction"] or ""
    exponent = match["exponent"] or ""
    canonical = (match["sign"] or "") + match["integer"].replace(",", "") + fraction + exponent
    value = Decimal(canonical.replace("\N{MINUS SIGN}", "-"))
    if abs(value.adjusted()) > _MAX_ADJUSTED_EXPONENT:
        raise ValueError(f"number out of range: {text!r}")
    place = int(value.as_tuple().exponent)
    return Literal(stripped, value, Decimal(5).scaleb(place - 1), not (fraction or exponent))
