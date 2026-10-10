from decimal import Decimal

import pytest

from numeric_claim.literals import parse_literal


@pytest.mark.parametrize(
    ("text", "value", "is_integer"),
    [
        ("264", Decimal(264), True),
        ("0", Decimal(0), True),
        ("007", Decimal(7), True),
        ("1,234,567", Decimal(1234567), True),
        ("999", Decimal(999), True),
        ("+7", Decimal(7), True),
        ("-5", Decimal(-5), True),
        ("\N{MINUS SIGN}5", Decimal(-5), True),
        ("0.5", Decimal("0.5"), False),
        ("1,000.25", Decimal("1000.25"), False),
        ("3.47e-08", Decimal("3.47e-8"), False),
        ("2E5", Decimal("2e5"), False),
        ("1.5e+3", Decimal("1500"), False),
        ("\N{MINUS SIGN}3.5e\N{MINUS SIGN}2", Decimal("-0.035"), False),
        ("5.0", Decimal("5.0"), False),
        ("  42 ", Decimal(42), True),
    ],
)
def test_parses_printed_numbers(text: str, value: Decimal, is_integer: bool) -> None:
    literal = parse_literal(text)
    assert literal.value == value
    assert literal.is_integer is is_integer


@pytest.mark.parametrize(
    ("text", "half_unit"),
    [
        ("3.47e-08", Decimal("5e-11")),
        ("2e-8", Decimal("5e-9")),
        ("264", Decimal("0.5")),
        ("0.50", Decimal("0.005")),
        ("5.0", Decimal("0.05")),
        ("1,000.5", Decimal("0.05")),
        ("1.5e3", Decimal("50")),
        ("1e1", Decimal("5")),
        ("12e0", Decimal("0.5")),
    ],
)
def test_half_unit_is_half_of_the_last_printed_digit(text: str, half_unit: Decimal) -> None:
    assert parse_literal(text).half_unit == half_unit


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "abc",
        "1.2.3",
        "--5",
        "1e",
        "1e+",
        "1,",
        ",5",
        ".5",
        "5.",
        "1,23",
        "12,34,567",
        "1,2345",
        "1,,000",
        "0x10",
        "1_000",
        "NaN",
        "inf",
        "1e5.5",
        "5 6",
        "5%",
        "1e99999999",
    ],
)
def test_rejects_text_that_is_not_a_number(text: str) -> None:
    with pytest.raises(ValueError, match="number"):
        parse_literal(text)


def test_integer_literal_has_no_decimal_point_and_no_exponent() -> None:
    assert parse_literal("1000").is_integer
    assert not parse_literal("1000.0").is_integer
    assert not parse_literal("1e3").is_integer
