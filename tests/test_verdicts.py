import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from helpers import DEFAULT_CONFIG, json_report, logging_claims, make_project, run
from numeric_claim.budget import Verdict, verdict
from numeric_claim.literals import parse_literal

NOISY = """
@claim("noisy", kind="seeded", atol=10.0, reference="1.5", jitter={"scale": 1e-6})
def noisy(*, scale=1.0):
    return 1.0 + (scale - 1.0) * 1e6
"""
QUIET = """
@claim("quiet", kind="seeded")
def quiet(*, x=1.5):
    return x
"""
PLAIN = """
@claim("plain", kind="seeded")
def plain():
    return 1.5
"""


def judged(
    relation: str,
    compared: str,
    printed: str,
    sigma: float,
    *,
    tolerance: float = 0.0,
    half_unit: str | None = None,
) -> Verdict:
    return verdict(
        relation,
        Decimal(compared),
        parse_literal(printed),
        tolerance,
        sigma,
        None if half_unit is None else Decimal(half_unit),
    )


@pytest.mark.parametrize(
    ("relation", "compared", "printed", "sigma", "label", "margin"),
    [
        ("lt", "0.5", "2", 0.25, "supported", "1.5"),
        ("lt", "0.5", "2", 1.0, "rounding-limited", "1.5"),
        ("lt", "2.5", "2", 0.25, "violated", "-0.5"),
        ("lt", "2", "2", 1.0, "violated", "0"),
        ("le", "0.5", "2", 0.25, "supported", "1.5"),
        ("le", "2", "2", 0.0, "rounding-limited", "0"),
        ("le", "2.5", "2", 5.0, "violated", "-0.5"),
        ("gt", "3.5", "2", 0.25, "supported", "1.5"),
        ("gt", "3.5", "2", 1.0, "rounding-limited", "1.5"),
        ("gt", "2", "2", 1.0, "violated", "0"),
        ("gt", "1.5", "2", 0.25, "violated", "-0.5"),
        ("ge", "3.5", "2", 0.25, "supported", "1.5"),
        ("ge", "2", "2", 0.0, "rounding-limited", "0"),
        ("ge", "1.5", "2", 5.0, "violated", "-0.5"),
        ("eq", "2.5", "2.5", 0.0078125, "supported", "0.05"),
        ("eq", "2.53125", "2.5", 0.0078125, "rounding-limited", "0.01875"),
        ("eq", "2.5625", "2.5", 0.0078125, "violated", "-0.0125"),
        ("eq", "2", "2", 0.0, "rounding-limited", "0"),
        ("eq", "2.25", "2", 5.0, "violated", "-0.25"),
    ],
)
def test_the_verdict_for_each_relation(
    relation: str, compared: str, printed: str, sigma: float, label: str, margin: str
) -> None:
    found = judged(relation, compared, printed, sigma)
    assert found.label == label
    assert found.margin == Decimal(margin)
    assert found.band == 3 * Decimal(sigma)


def test_a_margin_of_exactly_three_sigma_is_rounding_limited() -> None:
    exact = Verdict("rounding-limited", Decimal("1.5"), Decimal("1.5"))
    assert judged("lt", "0.5", "2", 0.5) == exact
    assert judged("lt", "0.5", "2", math.nextafter(0.5, 0.0)).label == "supported"
    assert judged("gt", "3.5", "2", 0.5).label == "rounding-limited"
    assert judged("gt", "3.5", "2", math.nextafter(0.5, 0.0)).label == "supported"
    assert judged("eq", "2.8", "2.5", 0.25, tolerance=1.0).label == "rounding-limited"
    assert judged("eq", "2.8", "2.5", math.nextafter(0.25, 0.0), tolerance=1.0).label == "supported"


def test_the_margin_and_the_band_are_exact() -> None:
    found = judged("lt", "21.6795939217990003556", "21.67959396", 1e-11)
    assert found.margin == Decimal("3.82009996444E-8")
    assert Fraction(found.band) == 3 * Fraction(1e-11)
    assert found.label == "supported"
    near = judged("eq", "2", "2.5", 0.25, tolerance=1.2)
    assert Fraction(near.margin) == Fraction("0.05") + Fraction(1.2) - Fraction("0.5")
    assert near.margin < near.band
    assert near.label == "rounding-limited"
    tiny = judged("gt", "1e-400", "0", 0.0)
    assert (tiny.label, tiny.margin) == ("supported", Decimal("1e-400"))


def test_eq_uses_the_allowance_of_the_relation() -> None:
    assert judged("eq", "2.5", "2.5", 0.0).margin == Decimal("0.05")
    assert judged("eq", "2.5", "2.5", 0.0, tolerance=0.25).margin == Decimal("0.3")
    assert judged("eq", "2.75", "2.5", 0.0, tolerance=0.25) == Verdict(
        "supported", Decimal("0.05"), Decimal(0)
    )
    assert judged("eq", "3", "3", 0.0, tolerance=0.25).margin == 0
    assert judged("eq", "3.125", "3", 0.0, tolerance=0.25).label == "violated"


@pytest.mark.parametrize(
    ("relation", "printed", "half_unit", "label"),
    [
        ("lt", "0.005", "0.005", "unsupported"),
        ("le", "0.005", "0.005", "unsupported"),
        ("lt", "0.004", "0.005", "unsupported"),
        ("le", "1e-9", "0.005", "unsupported"),
        ("lt", "0.006", "0.005", "supported"),
        ("le", "0.006", "0.005", "supported"),
        ("lt", "0.005", "0", "supported"),
        ("gt", "0.0001", "0.005", "supported"),
        ("ge", "0.0001", "0.005", "supported"),
        ("eq", "0.001", "0.005", "supported"),
    ],
)
def test_an_upper_bound_on_the_error_within_the_references_half_unit_is_unsupported(
    relation: str, printed: str, half_unit: str, label: str
) -> None:
    found = judged(relation, "0.001", printed, 0.0, half_unit=half_unit, tolerance=0.5)
    assert found.label == label


def test_unsupported_comes_before_violated_and_only_applies_to_the_error() -> None:
    assert judged("lt", "0.5", "0.005", 0.0, half_unit="0.005").label == "unsupported"
    assert judged("lt", "0.5", "0.006", 0.0, half_unit="0.005").label == "violated"
    assert judged("lt", "0.001", "0.005", 0.0).label == "supported"


def test_an_unknown_relation_is_rejected() -> None:
    with pytest.raises(ValueError, match="relation"):
        judged("ne", "1", "2", 0.0)


def sentinel(relation: str, literal: str, claim: str = "noisy") -> dict[str, str]:
    return {"README.md": f"<!-- claim: {claim} {relation} -->{literal}<!-- /claim -->\n"}


def error_binding(relation: str) -> str:
    return (
        DEFAULT_CONFIG + "[[bind]]\nfiles = ['NOTES.txt']\npattern = 'bound: (\\S+)'\n"
        f"claim = 'noisy'\nrelation = '{relation}'\nquantity = 'error'\n"
    )


def occurrence_of(
    root: Path, files: dict[str, str], config: str = DEFAULT_CONFIG
) -> tuple[int, dict[str, Any]]:
    make_project(root, logging_claims(NOISY, QUIET, PLAIN), files, config)
    assert run(root, "lock") == 0
    status, report = json_report(root, "check")
    assert [claim["status"] for claim in report["claims"]] == ["ok", "ok", "ok"]
    (occurrence,) = report["occurrences"]
    return status, occurrence


def assert_consistent(occurrence: dict[str, Any], verdict: str) -> None:
    assert occurrence["verdict"] == verdict
    assert occurrence["result"] == ("pass" if verdict == "supported" else "fail")
    margin, band = occurrence["margin"], occurrence["band"]
    assert isinstance(margin, float)
    assert isinstance(band, float)
    assert 1.0 < band < 2.5
    if verdict == "supported":
        assert margin > band
    if verdict == "rounding-limited":
        assert 0 <= margin <= band


@pytest.mark.parametrize(
    ("relation", "literal", "verdict"),
    [
        ("lt", "100", "supported"),
        ("le", "100", "supported"),
        ("lt", "1.5", "rounding-limited"),
        ("le", "1.0", "rounding-limited"),
        ("lt", "1.0", "violated"),
        ("le", "0.5", "violated"),
        ("gt", "-100", "supported"),
        ("ge", "-100", "supported"),
        ("gt", "0.5", "rounding-limited"),
        ("ge", "1.0", "rounding-limited"),
        ("gt", "1.0", "violated"),
        ("ge", "5", "violated"),
        ("eq", "1.0", "supported"),
        ("eq", "10.5", "rounding-limited"),
        ("eq", "20.0", "violated"),
    ],
)
def test_the_value_of_a_budgeted_claim_gets_a_verdict_and_only_supported_passes(
    tmp_path: Path, relation: str, literal: str, verdict: str
) -> None:
    status, occurrence = occurrence_of(tmp_path, sentinel(relation, literal))
    assert (occurrence["quantity"], occurrence["compared"]) == ("value", 1.0)
    assert_consistent(occurrence, verdict)
    assert status == (0 if verdict == "supported" else 1)


@pytest.mark.parametrize(
    ("relation", "literal", "verdict"),
    [
        ("lt", "100", "supported"),
        ("le", "100", "supported"),
        ("lt", "1", "rounding-limited"),
        ("le", "0.5", "rounding-limited"),
        ("lt", "0.5", "violated"),
        ("lt", "0.06", "violated"),
        ("lt", "0.05", "unsupported"),
        ("le", "0.05", "unsupported"),
        ("le", "0.01", "unsupported"),
        ("lt", "1e-9", "unsupported"),
        ("gt", "0.01", "rounding-limited"),
        ("ge", "0.5", "rounding-limited"),
        ("gt", "0.5", "violated"),
        ("eq", "0.5", "supported"),
        ("eq", "30.0", "violated"),
    ],
)
def test_the_error_of_a_budgeted_claim_gets_a_verdict_and_only_supported_passes(
    tmp_path: Path, relation: str, literal: str, verdict: str
) -> None:
    files = {"NOTES.txt": f"bound: {literal}\n"}
    status, occurrence = occurrence_of(tmp_path, files, error_binding(relation))
    assert (occurrence["quantity"], occurrence["compared"]) == ("error", 0.5)
    assert_consistent(occurrence, verdict)
    assert status == (0 if verdict == "supported" else 1)


def test_a_quiet_claim_supports_the_number_it_prints(tmp_path: Path) -> None:
    status, occurrence = occurrence_of(tmp_path, sentinel("eq", "1.5", "quiet"))
    assert status == 0
    assert (occurrence["verdict"], occurrence["result"]) == ("supported", "pass")
    assert 0.05 <= occurrence["margin"] < 0.0500001
    assert 0 < occurrence["band"] < 1e-14


def test_an_integer_literal_leaves_a_budgeted_claim_no_margin(tmp_path: Path) -> None:
    status, occurrence = occurrence_of(tmp_path, sentinel("eq", "1", "noisy"))
    assert status == 1
    assert (occurrence["verdict"], occurrence["margin"]) == ("rounding-limited", 0.0)


def test_a_claim_without_a_budget_gets_no_verdict(tmp_path: Path) -> None:
    status, occurrence = occurrence_of(tmp_path, sentinel("eq", "1.5", "plain"))
    assert status == 0
    assert (occurrence["result"], occurrence["quantity"], occurrence["compared"]) == (
        "pass",
        "value",
        1.5,
    )
    assert (occurrence["margin"], occurrence["band"], occurrence["verdict"]) == (None, None, None)


@pytest.mark.parametrize(
    ("document", "result"),
    [
        ("<!-- claim: noisy -->about one<!-- /claim -->\n", "bad-literal"),
        ("<!-- claim: nowhere -->1.0<!-- /claim -->\n", "unknown-claim"),
    ],
)
def test_an_occurrence_that_is_not_compared_gets_no_verdict(
    tmp_path: Path, document: str, result: str
) -> None:
    status, occurrence = occurrence_of(tmp_path, {"README.md": document})
    assert status == 1
    assert occurrence["result"] == result
    assert (occurrence["margin"], occurrence["band"], occurrence["verdict"]) == (None, None, None)


def test_the_text_report_shows_the_verdict_of_each_occurrence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = {
        "NOTES.txt": "bound: 0.05\n",
        "README.md": "<!-- claim: noisy lt -->100<!-- /claim -->\n"
        "<!-- claim: noisy gt -->0.5<!-- /claim -->\n"
        "<!-- claim: plain -->1.5<!-- /claim -->\n",
    }
    make_project(tmp_path, logging_claims(NOISY, QUIET, PLAIN), files, error_binding("lt"))
    assert run(tmp_path, "lock") == 0
    capsys.readouterr()
    status, report = json_report(tmp_path, "check")
    assert status == 1
    band = f"{report['occurrences'][0]['band']:.3g}"
    lines = capsys.readouterr().out.splitlines()
    assert lines[3:] == [
        "NOTES.txt:1 noisy lt 0.05 value=1.0 fail quantity=error compared=0.5 margin=-0.45 "
        f"3sigma={band} verdict=unsupported",
        "README.md:1 noisy lt 100 value=1.0 pass quantity=value compared=1.0 margin=99 "
        f"3sigma={band} verdict=supported",
        "README.md:2 noisy gt 0.5 value=1.0 fail quantity=value compared=1.0 margin=0.5 "
        f"3sigma={band} verdict=rounding-limited",
        "README.md:3 plain eq 1.5 value=1.5 pass",
        "3 claims (3 ok), 4 occurrences (2 fail, 2 pass): exit 1",
    ]
