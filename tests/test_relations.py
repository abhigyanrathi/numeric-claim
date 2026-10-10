import pytest

from numeric_claim.literals import parse_literal
from numeric_claim.relations import holds


def eq(value: int | float, text: str, tolerance: float = 0.0) -> bool:
    return holds("eq", value, parse_literal(text), tolerance)


def test_eq_with_an_integer_literal_is_exact() -> None:
    assert eq(264, "264")
    assert eq(264.0, "264")
    assert not eq(264, "265")
    assert not eq(264.0000001, "264")


def test_eq_with_an_integer_literal_ignores_the_claim_tolerance() -> None:
    assert not eq(264.5, "264", tolerance=10.0)


def test_eq_boundary_exactly_half_a_unit_passes_and_just_beyond_fails() -> None:
    assert eq(12.5, "12e0")
    assert eq(11.5, "12e0")
    assert not eq(12.500000000000002, "12e0")
    assert not eq(11.499999999999998, "12e0")


def test_eq_boundary_with_a_larger_unit() -> None:
    assert eq(15, "1e1")
    assert eq(5, "1e1")
    assert not eq(15.000000000000002, "1e1")
    assert not eq(4.999999999999999, "1e1")


def test_eq_with_decimals_uses_the_last_printed_digit() -> None:
    assert eq(3.4701e-08, "3.47e-08")
    assert eq(3.4749e-08, "3.47e-08")
    assert not eq(3.4751e-08, "3.47e-08")
    assert eq(2.4e-8, "2e-8")
    assert not eq(2.6e-8, "2e-8")


def test_eq_adds_the_claim_tolerance_to_the_half_unit() -> None:
    assert eq(12.75, "12e0", tolerance=0.25)
    assert not eq(12.76, "12e0", tolerance=0.25)


def test_eq_compares_the_exact_binary_value() -> None:
    assert eq(0.1 + 0.2, "0.3")
    assert eq(0.30000000000000004, "0.30")
    assert not eq(0.95, "1.0")
    assert eq(0.96, "1.0")


@pytest.mark.parametrize(
    ("relation", "value", "text", "expected"),
    [
        ("lt", 3, "4", True),
        ("lt", 4, "4", False),
        ("lt", 5, "4", False),
        ("le", 3, "4", True),
        ("le", 4, "4", True),
        ("le", 5, "4", False),
        ("gt", 5, "4", True),
        ("gt", 4, "4", False),
        ("gt", 3, "4", False),
        ("ge", 5, "4", True),
        ("ge", 4, "4", True),
        ("ge", 3, "4", False),
        ("lt", 3.5e-8, "4e-8", True),
        ("lt", 5.85e-8, "4e-8", False),
        ("ge", -1, "\N{MINUS SIGN}2", True),
    ],
)
def test_bounds_treat_the_printed_number_as_a_bound(
    relation: str, value: int | float, text: str, expected: bool
) -> None:
    assert holds(relation, value, parse_literal(text), 0.0) is expected


def test_bounds_ignore_the_claim_tolerance() -> None:
    assert not holds("lt", 4.0000001, parse_literal("4"), 1.0)
    assert not holds("gt", 3.9999999, parse_literal("4"), 1.0)


def test_unknown_relation_is_rejected() -> None:
    with pytest.raises(ValueError, match="relation"):
        holds("ne", 1, parse_literal("1"), 0.0)
