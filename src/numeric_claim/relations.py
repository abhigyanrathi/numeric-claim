"""How a computed value relates to the number a document prints."""

from decimal import Decimal
from fractions import Fraction

from numeric_claim.claim import Value
from numeric_claim.literals import Literal

RELATIONS = ("eq", "lt", "le", "gt", "ge")
QUANTITIES = ("value", "error")


def holds(relation: str, value: Value | Decimal, literal: Literal, tolerance: float) -> bool:
    """Whether ``value`` stands in ``relation`` to the printed number, compared exactly."""
    computed = Fraction(value)
    printed = Fraction(literal.value)
    match relation:
        case "eq":
            if literal.is_integer:
                return computed == printed
            limit = Fraction(literal.half_unit) + Fraction(tolerance)
            return abs(computed - printed) <= limit
        case "lt":
            return computed < printed
        case "le":
            return computed <= printed
        case "gt":
            return computed > printed
        case "ge":
            return computed >= printed
        case _:
            raise ValueError(f"unknown relation {relation!r}")
