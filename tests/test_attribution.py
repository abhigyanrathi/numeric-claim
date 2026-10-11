from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from helpers import json_report, logging_claims, make_project, run
from numeric_claim.budget import Attribution, Budget, Rung, attribute, dominant
from numeric_claim.literals import parse_literal

SERIES = """
@claim("series", kind="seeded", reference="2.0", ladders={"terms": [20, 40]})
def series(*, x=0.5, terms=10):
    return sum(x**k for k in range(terms))
"""
WOBBLY = """
@claim("wobbly", kind="seeded", reference="1.5", jitter={"scale": 1e-6})
def wobbly(*, scale=1.0):
    return 1.0 + (scale - 1.0) * 1e6
"""
THIRD = """
@claim("{name}", kind="seeded", reference="{reference}"{options})
def {name}(*, x=1.0, n=10):
    return x / 3.0 + (n - 10) * {slope}
"""
TIED = """
@claim("tied", kind="seeded", reference="1.0", ladders={"alpha": [2], "zeta": [2, 3]})
def tied(*, x=1.5, zeta=1, alpha=1):
    return x if zeta == 1 and alpha == 1 else x / 1.5
"""
NO_REFERENCE = """
@claim("free", kind="seeded", ladders={"n": [20]})
def free(*, x=1.5, n=10):
    return x / n
"""
NO_BUDGET = """
@claim("fixed", kind="seeded", reference="1.5")
def fixed():
    return 1.5
"""


def third(name: str, reference: str, options: str = "", slope: str = "0.0") -> str:
    return THIRD.format(name=name, reference=reference, options=options, slope=slope)


def label(error: str, gains: dict[str, str], sigma: float, half_unit: str = "0") -> str:
    exact = {name: Decimal(gain) for name, gain in gains.items()}
    return dominant(Decimal(error), exact, sigma, Decimal(half_unit))


def test_a_ladder_that_removes_half_the_error_and_more_than_three_sigma_dominates() -> None:
    assert label("1", {"n": "0.75"}, 0.125) == "n"
    assert label("1", {"n": "0.5"}, 0.125) == "n"
    assert label("1", {"n": "1"}, 0.0) == "n"
    assert label("1", {"n": "0.9"}, 0.25) == "n"


def test_a_ladder_gain_below_half_the_error_or_within_three_sigma_does_not_dominate() -> None:
    assert label("1", {"n": "0.4999"}, 0.125) == "unexplained"
    assert label("1", {"n": "0.75"}, 0.25) == "rounding"
    assert label("1", {"n": "0"}, 0.0) == "unexplained"
    assert label("1", {"n": "-0.5"}, 0.0) == "unexplained"


def test_the_ladder_with_the_largest_gain_is_named_and_a_tie_goes_to_the_first() -> None:
    assert label("1", {"n": "0.6", "l_trunc": "0.9"}, 0.0) == "l_trunc"
    assert label("1", {"n": "0.9", "l_trunc": "0.6"}, 0.0) == "n"
    assert label("1", {"zeta": "0.75", "alpha": "0.75"}, 0.0) == "zeta"
    assert label("1", {"alpha": "0.75", "zeta": "0.75"}, 0.0) == "alpha"
    assert label("1", {"n": "-2", "l_trunc": "0.9"}, 0.0) == "l_trunc"


def test_rounding_dominates_when_three_sigma_is_at_least_half_the_error() -> None:
    assert label("1.5", {}, 0.25) == "rounding"
    assert label("1.5000001", {}, 0.25) == "unexplained"
    assert label("1", {}, 1.0, half_unit="5") == "rounding"
    assert label("0", {"n": "0"}, 0.0) == "rounding"


def test_the_reference_dominates_an_error_within_its_half_unit() -> None:
    assert label("1", {}, 0.0, half_unit="1") == "reference"
    assert label("1", {}, 0.0, half_unit="0.9999") == "unexplained"
    assert label("1", {"n": "0.4999"}, 0.125, half_unit="1") == "reference"


def test_an_error_nothing_accounts_for_is_unexplained() -> None:
    assert label("1", {}, 0.0) == "unexplained"
    assert label("1", {"n": "0.25", "l_trunc": "-3"}, 0.125, half_unit="0.5") == "unexplained"


def test_attribute_measures_the_value_and_every_ladder_against_the_reference() -> None:
    budget = Budget(
        1e-15,
        None,
        {
            "terms": (Rung(20, 2 - 2.0**-19), Rung(40, 2 - 2.0**-39)),
            "width": (Rung(11.0, 1.5), Rung(12.0, 2.25)),
        },
    )
    found = attribute(2 - 2.0**-9, budget, parse_literal("2.0"))
    assert isinstance(found, Attribution)
    assert found.error == Decimal(2) ** -9
    assert found.ladders == {
        "terms": (Decimal(2) ** -19, Decimal(2) ** -39),
        "width": (Decimal("0.5"), Decimal("0.25")),
    }
    assert {name: Fraction(gain) for name, gain in found.gains.items()} == {
        "terms": Fraction(1, 2**9) - Fraction(1, 2**39),
        "width": Fraction(1, 2**9) - Fraction(1, 4),
    }
    assert list(found.gains) == ["terms", "width"]
    assert found.label == "terms"


def test_attribute_keeps_digits_a_rounded_subtraction_would_lose() -> None:
    value, reference = 21.679593921799, parse_literal("21.679593920")
    found = attribute(value, Budget(1e-11, None, {}), reference)
    assert Fraction(found.error) == Fraction(value) - Fraction("21.679593920")
    assert len(found.error.as_tuple().digits) > 28
    assert found.error != Decimal(value) - Decimal("21.679593920")
    assert (found.ladders, found.gains, found.label) == ({}, {}, "unexplained")


def budgets(root: Path, *definitions: str) -> tuple[int, dict[str, dict[str, Any]]]:
    make_project(root, logging_claims(*definitions))
    assert run(root, "lock") == 0
    status, report = json_report(root, "check")
    assert all(claim["source"] == "lock" for claim in report["claims"])
    return status, {claim["name"]: claim["budget"] for claim in report["claims"]}


def test_each_label_through_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, found = budgets(
        tmp_path, SERIES, WOBBLY, third("coarse", "0.33"), third("fine", "0.3300")
    )
    assert status == 0
    assert {name: budget["attribution"] for name, budget in found.items()} == {
        "series": "terms",
        "wobbly": "rounding",
        "coarse": "reference",
        "fine": "unexplained",
    }
    series = found["series"]
    assert series["error"] == 2.0**-9
    assert series["gains"] == {"terms": 2.0**-9 - 2.0**-39}
    assert [(rung["setting"], rung["error"]) for rung in series["ladders"]["terms"]] == [
        (20, 2.0**-19),
        (40, 2.0**-39),
    ]
    assert found["wobbly"]["error"] == 0.5
    assert found["wobbly"]["gains"] == {}
    assert 3 * found["wobbly"]["sigma"] >= 0.25
    assert 0.0033 < found["coarse"]["error"] == found["fine"]["error"] < 0.0034
    output = capsys.readouterr().out
    assert "error=0.00195 attribution=terms" in output
    assert "error=0.5 attribution=rounding" in output
    assert "error=0.00333 attribution=reference" in output
    assert "error=0.00333 attribution=unexplained" in output


@pytest.mark.parametrize(
    ("slope", "attribution"),
    [("0.0", "unexplained"), ("0.001", "unexplained"), ("-0.0003", "n")],
)
def test_a_ladder_dominates_only_if_it_reduces_the_error(
    tmp_path: Path, slope: str, attribution: str
) -> None:
    _, found = budgets(tmp_path, third("third", "0.3300", ', ladders={"n": [15, 20]}', slope))
    budget = found["third"]
    assert budget["attribution"] == attribution
    errors = [rung["error"] for rung in budget["ladders"]["n"]]
    assert budget["gains"] == {"n": pytest.approx(budget["error"] - min(errors), abs=1e-15)}
    assert (budget["gains"]["n"] > 0) == (attribution == "n")


@pytest.mark.parametrize("arguments", [[], ["--recompute"]])
def test_a_tie_goes_to_the_first_ladder_in_signature_order(
    tmp_path: Path, arguments: list[str]
) -> None:
    make_project(tmp_path, logging_claims(TIED))
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check", *arguments)
    assert status == 0
    budget = report["claims"][0]["budget"]
    assert budget["gains"] == {"alpha": 0.5, "zeta": 0.5}
    assert budget["attribution"] == "zeta"
    assert budget["ladders"] == {
        "alpha": [{"setting": 2, "value": 1.0, "error": 0.0}],
        "zeta": [
            {"setting": 2, "value": 1.0, "error": 0.0},
            {"setting": 3, "value": 1.0, "error": 0.0},
        ],
    }


def test_a_claim_without_a_reference_or_without_a_budget_has_no_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, found = budgets(tmp_path, NO_REFERENCE, NO_BUDGET)
    assert status == 0
    assert found["fixed"] is None
    free = found["free"]
    assert (free["error"], free["gains"], free["attribution"]) == (None, None, None)
    assert free["ladders"] == {"n": [{"setting": 20, "value": 0.075, "error": None}]}
    assert free["sigma"] == free["nudge"] > 0
    assert free["jitter"] is None
    output = capsys.readouterr().out
    assert "attribution" not in output
    assert "claim fixed [seeded] ok value=1.5 (lock)\n" in output


def test_the_label_never_changes_the_exit_status(tmp_path: Path) -> None:
    files = {"README.md": "<!-- claim: fine gt -->0.3<!-- /claim -->\n"}
    make_project(tmp_path, logging_claims(third("fine", "0.3300")), files)
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    assert report["claims"][0]["budget"]["attribution"] == "unexplained"
    assert report["occurrences"][0]["verdict"] == "supported"
    assert status == 0
