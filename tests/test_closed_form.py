import math
import os
import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath
import pytest

import numeric_claim
from helpers import DEFAULT_CONFIG, calls, json_report, logged, logging_claims, make_project, run
from numeric_claim import closed_form
from numeric_claim.claim import ClosedForm
from numeric_claim.engine import resolve
from numeric_claim.errors import ClaimError
from numeric_claim.literals import Literal, parse_literal

SOURCE = Path(__file__).resolve().parents[1] / "src"
SQRT2 = "1.4142135623730950488016887242096980785696718753769"
ERROR = float(abs(Fraction(math.sqrt(2)) - Fraction(SQRT2)))
BOUND = (
    DEFAULT_CONFIG + "[[bind]]\nfiles = ['NOTES.txt']\npattern = 'within (\\S+)'\n"
    "claim = 'root'\nrelation = 'lt'\nquantity = 'error'\n"
)
FORM = """
def evaluate(mp):
    log("forms.log")
    return {expression}


@claim("root", kind="seeded", reference=closed_form(evaluate))
def root({parameters}):
    log()
    return math.sqrt({argument})
"""


def form(expression: str = "mp.sqrt(2)", *, budgeted: bool = True) -> str:
    parameters, argument = ("*, x=2.0", "x") if budgeted else ("", "2")
    source = FORM.format(expression=expression, parameters=parameters, argument=argument)
    return logging_claims(source)


def evaluations(root: Path) -> int:
    return len(logged(root, "forms.log"))


def test_closed_form_is_exported_and_returns_a_reference_to_declare() -> None:
    def pi(mp: Any) -> Any:
        return mp.pi

    assert numeric_claim.closed_form is closed_form
    assert "closed_form" in numeric_claim.__all__
    assert closed_form(pi) == ClosedForm(pi, 50)
    assert closed_form(pi, dps=20) == ClosedForm(pi, 20)
    assert closed_form(pi).text == "closed form"


@pytest.mark.parametrize("dps", [19, 0, -50, 20.0, 50.5, "50", None, True])
def test_dps_must_be_an_int_of_at_least_20(dps: Any) -> None:
    with pytest.raises(ValueError, match="dps must be an int of at least 20"):
        closed_form(lambda mp: mp.pi, dps=dps)


def test_the_reference_has_the_requested_precision() -> None:
    fifty = resolve(closed_form(lambda mp: mp.sqrt(2)))
    assert fifty == Literal("closed form", Decimal(SQRT2), Decimal(0), False)
    assert len(fifty.value.as_tuple().digits) == 50
    thirty = resolve(closed_form(lambda mp: mp.sqrt(2), dps=30))
    assert thirty.value == Decimal("1.41421356237309504880168872421")
    assert fifty.value != Decimal(math.sqrt(2))
    assert 9.6e-17 < abs(Decimal(math.sqrt(2)) - fifty.value) < 9.7e-17


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda mp: 3, "3"),
        (lambda mp: 0.5, "0.5"),
        (lambda mp: "0.1", "0.1"),
        (lambda mp: 0.1, "0.1000000000000000055511151231257827021181583404541"),
        (lambda mp: mp.mpf(1) / 4, "0.25"),
        (lambda mp: -(mp.mpf(10) ** 100), "-1e100"),
        (lambda mp: mp.mpf("1e-30"), "1e-30"),
        (lambda mp: mp.mpf(0), "0"),
    ],
)
def test_a_closed_form_may_return_any_number_mpmath_reads(
    function: Callable[[Any], Any], expected: str
) -> None:
    reference = resolve(closed_form(function))
    assert reference.value == Decimal(expected)
    assert reference.half_unit == 0
    assert reference.text == "closed form"


def test_a_literal_reference_resolves_to_itself() -> None:
    literal = parse_literal("21.679593920")
    assert resolve(literal) is literal


def test_mpmath_precision_is_restored_afterwards() -> None:
    before = mpmath.mp.dps
    seen: list[int] = []

    def function(mp: Any) -> Any:
        seen.append(mp.mp.dps)
        return mp.sqrt(2)

    resolve(closed_form(function, dps=40))
    assert seen == [40]
    assert mpmath.mp.dps == before
    with pytest.raises(ClaimError):
        resolve(closed_form(lambda mp: 1 / 0, dps=40))
    assert mpmath.mp.dps == before


def leave(mp: Any) -> Any:
    raise SystemExit(3)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (lambda mp: 1 / 0, "the closed-form reference failed: ZeroDivisionError: division by zero"),
        (leave, "the closed-form reference failed: SystemExit: 3"),
        (lambda mp: mp.inf, "the closed-form reference failed: ValueError: not a number: 'inf'"),
        (lambda mp: mp.nan, "the closed-form reference failed: ValueError: not a number: 'nan'"),
        (lambda mp: mp.mpf(10) ** 20000, "the closed-form reference failed: ValueError: number"),
        (lambda mp: None, "the closed-form reference failed: TypeError"),
        (lambda mp: mp.mpc(1, 1), "the closed-form reference failed: TypeError"),
        (lambda mp: "about three", "the closed-form reference failed: ValueError"),
    ],
)
def test_a_closed_form_that_gives_no_finite_number_is_a_claim_error(
    function: Callable[[Any], Any], message: str
) -> None:
    with pytest.raises(ClaimError) as raised:
        resolve(closed_form(function))
    assert str(raised.value).startswith(message)


def test_a_closed_form_without_mpmath_is_a_claim_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mpmath", None)
    with pytest.raises(ClaimError, match="needs mpmath, which is not installed"):
        resolve(closed_form(lambda mp: mp.sqrt(2)))


def test_a_closed_form_is_evaluated_once_per_run_and_only_by_check(tmp_path: Path) -> None:
    notes = {"NOTES.txt": "within 1e-9\nwithin 1e-8\n"}
    make_project(tmp_path, form(), notes, BOUND)
    assert run(tmp_path, "list") == 0
    assert run(tmp_path, "lock") == 0
    assert evaluations(tmp_path) == 0
    status, report = json_report(tmp_path, "check")
    assert status == 0
    assert evaluations(tmp_path) == 1
    claim = report["claims"][0]
    assert claim["reference"] == "closed form"
    assert claim["budget"]["error"] == ERROR
    assert claim["budget"]["attribution"] == "rounding"
    assert [(o["compared"], o["verdict"]) for o in report["occurrences"]] == [
        (ERROR, "supported"),
        (ERROR, "supported"),
    ]
    assert run(tmp_path, "check", "--recompute") == 0
    assert evaluations(tmp_path) == 2


def test_a_closed_form_no_document_or_budget_needs_is_not_evaluated(tmp_path: Path) -> None:
    make_project(tmp_path, form(budgeted=False), {"NOTES.txt": "nothing here\n"}, BOUND)
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check", "--recompute")
    assert status == 0
    assert evaluations(tmp_path) == 0
    claim = report["claims"][0]
    assert (claim["reference"], claim["budget"]) == ("closed form", None)


def test_an_error_binding_needs_the_closed_form_of_a_claim_without_a_budget(
    tmp_path: Path,
) -> None:
    make_project(tmp_path, form(budgeted=False), {"NOTES.txt": "within 1e-16\n"}, BOUND)
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    assert status == 0
    assert evaluations(tmp_path) == 1
    assert [(o["compared"], o["result"], o["verdict"]) for o in report["occurrences"]] == [
        (ERROR, "pass", None)
    ]


def test_a_claim_whose_closed_form_raises_is_in_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_project(tmp_path, form("1 / 0"), {"NOTES.txt": "within 1e-9\n"}, BOUND)
    assert run(tmp_path, "lock") == 0
    assert calls(tmp_path) == 33
    capsys.readouterr()
    status, report = json_report(tmp_path, "check")
    assert status == 1
    assert calls(tmp_path) == 33
    note = "the closed-form reference failed: ZeroDivisionError: division by zero"
    claim = report["claims"][0]
    assert (claim["status"], claim["note"], claim["reference"]) == ("error", note, "closed form")
    assert claim["budget"] is None
    assert [o["result"] for o in report["occurrences"]] == ["claim-error"]
    assert f"claim root [seeded] error value=-: {note}" in capsys.readouterr().out


def test_a_claim_whose_closed_form_needs_a_missing_mpmath_is_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_project(tmp_path, form(), {"NOTES.txt": "within 1e-9\n"}, BOUND)
    assert run(tmp_path, "lock") == 0
    monkeypatch.setitem(sys.modules, "mpmath", None)
    status, report = json_report(tmp_path, "check")
    assert status == 1
    assert report["claims"][0]["status"] == "error"
    assert report["claims"][0]["note"] == (
        "the closed-form reference needs mpmath, which is not installed"
    )
    assert evaluations(tmp_path) == 0


def test_the_package_imports_without_numpy_or_mpmath() -> None:
    code = (
        "import sys\n"
        "sys.modules['numpy'] = None\n"
        "sys.modules['mpmath'] = None\n"
        "import numeric_claim.cli\n"
        "print(sorted(numeric_claim.__all__))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SOURCE), environment.get("PYTHONPATH")])
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "['ClaimContext', 'claim', 'closed_form']"
