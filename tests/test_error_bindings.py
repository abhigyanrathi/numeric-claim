import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from helpers import DEFAULT_CONFIG, json_report, logging_claims, make_project, run
from numeric_claim.claim import ClaimSpec
from numeric_claim.config import load_config
from numeric_claim.engine import ClaimOutcome, judge_all
from numeric_claim.errors import ConfigError
from numeric_claim.loader import LoadedClaim
from numeric_claim.scan import Occurrence, collect

ROOT = math.sqrt(2)
SQRT2 = "1.4142135623730950488016887242096980785696718753769"
ERROR = float(abs(Fraction(ROOT) - Fraction("1.4142")))
CLAIM = """
@claim("root", kind="seeded", reference={reference}{options})
def root():
    return math.sqrt(2)
"""
BARE = """
@claim("bare", kind="seeded")
def bare():
    return 1.0
"""


def claims(reference: str = '"1.4142"', options: str = "") -> str:
    return logging_claims(CLAIM.format(reference=reference, options=options), BARE)


def bound(claim: str = '"root"', relation: str = "lt", quantity: str = 'quantity = "error"') -> str:
    return (
        DEFAULT_CONFIG + "[[bind]]\nfiles = ['NOTES.txt']\npattern = 'within (\\S+)'\n"
        f"claim = {claim}\nrelation = '{relation}'\n{quantity}\n"
    )


def checked(root: Path, source: str, notes: str, config: str) -> tuple[int, dict[str, Any]]:
    make_project(root, source, {"NOTES.txt": notes}, config)
    assert run(root, "lock") == 0
    return json_report(root, "check")


def test_the_error_is_the_distance_from_the_value_to_a_literal_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, report = checked(tmp_path, claims(), "within 2e-5\nwithin 1e-5\n", bound())
    assert status == 1
    assert 1.3562e-5 < ERROR < 1.3563e-5
    assert [
        (o["line"], o["claim"], o["quantity"], o["compared"], o["value"], o["result"])
        for o in report["occurrences"]
    ] == [
        (1, "root", "error", ERROR, ROOT, "pass"),
        (2, "root", "error", ERROR, ROOT, "fail"),
    ]
    assert all(
        (o["margin"], o["band"], o["verdict"]) == (None, None, None) for o in report["occurrences"]
    )
    root = report["claims"][1]
    assert (root["name"], root["reference"], root["budget"]) == ("root", "1.4142", None)
    lines = capsys.readouterr().out.splitlines()
    assert lines[-3:-1] == [
        f"NOTES.txt:1 root lt 2e-5 value={ROOT!r} pass quantity=error compared=1.36e-05",
        f"NOTES.txt:2 root lt 1e-5 value={ROOT!r} fail quantity=error compared=1.36e-05",
    ]


def test_the_error_against_a_closed_form_is_measured_beyond_float_precision(tmp_path: Path) -> None:
    source = claims(reference="closed_form(lambda mp: mp.sqrt(2))")
    status, report = checked(tmp_path, source, "within 1e-16\nwithin 9e-17\n", bound())
    assert status == 1
    expected = float(abs(Fraction(ROOT) - Fraction(SQRT2)))
    assert 9.6e-17 < expected < 9.7e-17
    assert [(o["compared"], o["result"]) for o in report["occurrences"]] == [
        (expected, "pass"),
        (expected, "fail"),
    ]
    assert report["claims"][1]["reference"] == "closed form"


@pytest.mark.parametrize(
    ("options", "literal", "result"),
    [
        ("", "1.36e-5", "pass"),
        ("", "1.35e-5", "fail"),
        ("", "1.30e-5", "fail"),
        (", atol=1e-6", "1.30e-5", "pass"),
        (", atol=1e-6", "1.20e-5", "fail"),
        (", rtol=1e-6", "1.30e-5", "pass"),
        (", rtol=1e-7", "1.30e-5", "fail"),
    ],
)
def test_eq_allows_the_error_half_a_unit_plus_the_tolerance_at_the_claims_value(
    tmp_path: Path, options: str, literal: str, result: str
) -> None:
    config = bound(relation="eq")
    status, report = checked(tmp_path, claims(options=options), f"within {literal}\n", config)
    assert [o["result"] for o in report["occurrences"]] == [result]
    assert status == (0 if result == "pass" else 1)


@pytest.mark.parametrize(
    ("relation", "literal", "result"),
    [
        ("le", "2e-5", "pass"),
        ("gt", "1e-5", "pass"),
        ("ge", "2e-5", "fail"),
        ("gt", "1", "fail"),
    ],
)
def test_every_relation_applies_to_the_error(
    tmp_path: Path, relation: str, literal: str, result: str
) -> None:
    config = bound(relation=relation)
    _, report = checked(tmp_path, claims(), f"within {literal}\n", config)
    assert [o["result"] for o in report["occurrences"]] == [result]


def test_sentinels_macros_and_plain_bindings_compare_the_value(tmp_path: Path) -> None:
    files = {
        "NOTES.txt": "within 1.4142\n",
        "README.md": "<!-- claim: root -->1.4142<!-- /claim -->\n",
        "paper.tex": "\\claim{root}{1.4142}\n",
    }
    make_project(tmp_path, claims(), files, bound(relation="eq", quantity=""))
    found = collect(load_config(tmp_path / "numeric-claim.toml"))
    assert [(occurrence.file, occurrence.quantity) for occurrence in found] == [
        ("NOTES.txt", "value"),
        ("README.md", "value"),
        ("paper.tex", "value"),
    ]
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    assert status == 0
    assert [(o["quantity"], o["compared"], o["result"]) for o in report["occurrences"]] == [
        ("value", ROOT, "pass")
    ] * 3


def test_an_error_binding_gives_error_occurrences(tmp_path: Path) -> None:
    make_project(tmp_path, claims(), {"NOTES.txt": "within 2e-5\n"}, bound(claim='["root"]'))
    assert collect(load_config(tmp_path / "numeric-claim.toml")) == [
        Occurrence("NOTES.txt", 1, "root", "lt", "2e-5", "pattern", "error")
    ]


@pytest.mark.parametrize(
    ("quantity", "message"),
    [
        ('quantity = "relative"', "[[bind]] entry 1: unknown quantity 'relative'; use one of"),
        ('quantity = "Error"', "unknown quantity 'Error'"),
        ('quantity = ""', "unknown quantity ''"),
        ("quantity = 3", "unknown quantity 3"),
        ('quantity = ["error"]', "unknown quantity ['error']"),
    ],
)
def test_any_other_quantity_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantity: str, message: str
) -> None:
    make_project(tmp_path, claims(), {"NOTES.txt": "within 2e-5\n"}, bound(quantity=quantity))
    assert run(tmp_path, "check") == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("command", ["check", "lock", "list"])
@pytest.mark.parametrize("names", ['"bare"', '["root", "bare"]'])
def test_an_error_binding_to_a_claim_without_a_reference_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, names: str
) -> None:
    make_project(tmp_path, claims(), {"NOTES.txt": "nothing bound here\n"}, bound(claim=names))
    assert run(tmp_path, command) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "[[bind]] entry 1: quantity 'error' needs a reference, and claim 'bare' declares none"
        in captured.err
    )
    assert not (tmp_path / "claims.lock.json").exists()


def test_a_value_binding_needs_no_reference(tmp_path: Path) -> None:
    config = bound(claim='"bare"', relation="eq", quantity='quantity = "value"')
    status, report = checked(tmp_path, claims(), "within 1.0\n", config)
    assert status == 0
    assert [(o["quantity"], o["result"]) for o in report["occurrences"]] == [("value", "pass")]


def test_an_error_binding_to_a_name_that_matches_no_claim_is_an_unknown_claim(
    tmp_path: Path,
) -> None:
    config = bound(claim='["root", "nowhere"]')
    status, report = checked(tmp_path, claims(), "within 2e-5\n", config)
    assert status == 1
    assert [(o["claim"], o["result"], o["compared"]) for o in report["occurrences"]] == [
        ("root", "pass", ERROR),
        ("nowhere", "unknown-claim", None),
    ]


def test_the_error_of_a_claim_in_error_is_a_claim_error(tmp_path: Path) -> None:
    broken = """
    @claim("root", kind="seeded", reference="1.4142")
    def root():
        raise RuntimeError("boom")
    """
    make_project(tmp_path, logging_claims(broken), {"NOTES.txt": "within 2e-5\n"}, bound())
    status, report = json_report(tmp_path, "check")
    assert status == 1
    assert [(o["result"], o["compared"]) for o in report["occurrences"]] == [("claim-error", None)]


def test_a_bad_literal_still_reports_the_compared_error(tmp_path: Path) -> None:
    status, report = checked(tmp_path, claims(), "within reach\n", bound())
    assert status == 1
    assert [(o["result"], o["compared"]) for o in report["occurrences"]] == [("bad-literal", ERROR)]


def test_judging_an_error_without_a_resolved_reference_is_a_configuration_error() -> None:
    loaded = LoadedClaim(ClaimSpec("c", "seeded", 1e-12, 0.0, (), ()), lambda: 1.0, Path("c.py"))
    outcome = ClaimOutcome("c", "seeded", "ok", 1.0, 1.0, "lock", "")
    occurrence = Occurrence("NOTES.txt", 3, "c", "lt", "1e-8", "pattern", "error")
    with pytest.raises(ConfigError, match=r"NOTES\.txt:3: the error of claim 'c' needs a ref"):
        judge_all([occurrence], [loaded], [outcome])
