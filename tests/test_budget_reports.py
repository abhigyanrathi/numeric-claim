import json
import math
from pathlib import Path
from typing import Any

import pytest

from helpers import DEFAULT_CONFIG, calls, json_report, logged, logging_claims, make_project, run

VALUE = 2.0 + 1.0 / 10 + 10.0 / 7.0
ROOT = math.sqrt(2.0)
TOTAL = """
@claim(
    "total",
    kind="seeded",
    reference="3.5",
    ladders={"width": [11, 12.5], "n": [20, 40]},
    jitter={"width": 1e-9},
)
def total(*, x=2.0, n=10, width=10.0):
    log(x=x, n=n, width=width)
    return x + 1.0 / n + width / 7.0
"""
FORMED = """
def evaluate(mp):
    log("forms.log")
    return mp.sqrt(2)


@claim("formed", kind="seeded", reference=closed_form(evaluate), packages=["json"], paths=["docs"])
def formed(*, zero=0.0, x=2.0, y=3.0):
    log()
    return math.sqrt(x) + zero * y
"""
COUNT = """
@claim("count")
def count():
    log()
    return 3
"""
CONFIG = (
    DEFAULT_CONFIG + "[[bind]]\nfiles = ['NOTES.txt']\npattern = 'within (\\S+)'\n"
    "claim = ['total', 'formed']\nrelation = 'lt'\nquantity = 'error'\n"
)
FILES = {
    "NOTES.txt": "within 0.1\n",
    "README.md": "<!-- claim: total ge -->3.5<!-- /claim -->\n"
    "<!-- claim: count -->3<!-- /claim -->\n",
    "docs/guide.md": "No numbers here.\n",
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    make_project(tmp_path, logging_claims(TOTAL, FORMED, COUNT), FILES, CONFIG)
    return tmp_path


def no_constant(name: str) -> Any:
    raise AssertionError(f"{name} is not JSON")


def rung(setting: float, value: float, error: float) -> dict[str, Any]:
    return {"setting": setting, "value": value, "error": pytest.approx(error)}


def test_list_shows_references_probes_and_quantities_without_computing(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, report = json_report(project, "list")
    assert status == 0
    assert calls(project) == 0
    assert logged(project, "forms.log") == []
    assert capsys.readouterr().out.splitlines() == [
        "claim count [exact] packages=- paths=- reference=- nudged=- ladders=- jitter=-",
        "claim formed [seeded] packages=json paths=docs reference=closed form nudged=x,y "
        "ladders=- jitter=-",
        "claim total [seeded] packages=- paths=- reference=3.5 nudged=x "
        "ladders=n:20,40;width:11,12.5 jitter=width:1e-09",
        "NOTES.txt:1 total lt 0.1 quantity=error",
        "NOTES.txt:1 formed lt 0.1 quantity=error",
        "README.md:1 total ge 3.5 quantity=value",
        "README.md:2 count eq 3 quantity=value",
        "3 claims, 4 occurrences",
    ]
    assert report["claims"] == [
        {
            "name": "count",
            "kind": "exact",
            "packages": [],
            "paths": [],
            "reference": None,
            "nudged": [],
            "ladders": {},
            "jitter": {},
        },
        {
            "name": "formed",
            "kind": "seeded",
            "packages": ["json"],
            "paths": ["docs"],
            "reference": "closed form",
            "nudged": ["x", "y"],
            "ladders": {},
            "jitter": {},
        },
        {
            "name": "total",
            "kind": "seeded",
            "packages": [],
            "paths": [],
            "reference": "3.5",
            "nudged": ["x"],
            "ladders": {"n": [20, 40], "width": [11, 12.5]},
            "jitter": {"width": 1e-09},
        },
    ]
    assert [(o["claim"], o["quantity"]) for o in report["occurrences"]] == [
        ("total", "error"),
        ("formed", "error"),
        ("total", "value"),
        ("count", "value"),
    ]
    assert not (project / "claims.lock.json").exists()


def test_the_json_report_describes_each_budget_and_verdict(project: Path) -> None:
    assert run(project, "lock") == 0
    status, report = json_report(project, "check")
    assert status == 0
    count, formed, total = report["claims"]
    assert (count["reference"], count["budget"]) == (None, None)
    assert formed["reference"] == "closed form"
    assert sorted(total) == [
        "budget",
        "kind",
        "locked_value",
        "name",
        "note",
        "random",
        "reference",
        "source",
        "status",
        "value",
    ]
    assert (total["reference"], total["value"]) == ("3.5", VALUE)
    budget = total["budget"]
    assert sorted(budget) == [
        "attribution",
        "error",
        "gains",
        "jitter",
        "ladders",
        "nudge",
        "sigma",
    ]
    assert budget["sigma"] == budget["jitter"] > budget["nudge"] > 0
    assert budget["error"] == pytest.approx(VALUE - 3.5, abs=1e-15)
    assert budget["attribution"] == "reference"
    assert budget["ladders"] == {
        "n": [
            rung(20, 2.0 + 1.0 / 20 + 10.0 / 7.0, 0.0214285714),
            rung(40, 2.0 + 1.0 / 40 + 10.0 / 7.0, 0.0464285714),
        ],
        "width": [
            rung(11, 2.0 + 1.0 / 10 + 11.0 / 7.0, 0.1714285714),
            rung(12.5, 2.0 + 1.0 / 10 + 12.5 / 7.0, 0.3857142857),
        ],
    }
    assert [type(rung["setting"]) for rung in budget["ladders"]["width"]] == [int, float]
    assert budget["gains"] == {
        "n": pytest.approx(0.0071428571),
        "width": pytest.approx(-0.1428571429),
    }
    assert all(
        isinstance(number, float)
        for number in (budget["nudge"], budget["jitter"], budget["sigma"], budget["error"])
    )
    first = report["occurrences"][0]
    assert sorted(first) == [
        "band",
        "claim",
        "compared",
        "file",
        "line",
        "literal",
        "margin",
        "note",
        "origin",
        "p",
        "p_holm",
        "quantity",
        "relation",
        "result",
        "t",
        "value",
        "verdict",
    ]
    assert (first["claim"], first["quantity"], first["verdict"]) == ("total", "error", "supported")
    assert first["compared"] == budget["error"]
    assert first["margin"] == pytest.approx(0.1 - (VALUE - 3.5), abs=1e-15)
    assert first["band"] == pytest.approx(3 * budget["sigma"], rel=1e-12)
    exact = report["occurrences"][3]
    assert (exact["claim"], exact["value"], exact["compared"]) == ("count", 3, 3.0)
    assert isinstance(exact["compared"], float)
    assert (exact["margin"], exact["band"], exact["verdict"]) == (None, None, None)


def test_the_text_report_shows_each_budget(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(project, "lock") == 0
    capsys.readouterr()
    status, report = json_report(project, "check")
    assert status == 0
    formed, total = (claim["budget"] for claim in report["claims"][1:])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "claim count [exact] ok value=3 (lock)",
        f"claim formed [seeded] ok value={ROOT!r} (lock) "
        f"sigma={formed['sigma']:.3g} (nudge={formed['nudge']:.3g}, jitter=-) "
        f"error={formed['error']:.3g} attribution={formed['attribution']}",
        f"claim total [seeded] ok value={VALUE!r} (lock) "
        f"sigma={total['sigma']:.3g} (nudge={total['nudge']:.3g}, jitter={total['jitter']:.3g}) "
        "error=0.0286 attribution=reference",
    ]
    band = f"{3 * total['sigma']:.3g}"
    assert lines[3] == (
        f"NOTES.txt:1 total lt 0.1 value={VALUE!r} pass quantity=error compared=0.0286 "
        f"margin=0.0714 3sigma={band} verdict=supported"
    )
    assert lines[5] == (
        f"README.md:1 total ge 3.5 value={VALUE!r} pass quantity=value compared={VALUE!r} "
        f"margin=0.0286 3sigma={band} verdict=supported"
    )
    assert lines[6:] == [
        "README.md:2 count eq 3 value=3 pass",
        "3 claims (3 ok), 4 occurrences (4 pass): exit 0",
    ]


def test_a_number_beyond_a_floats_range_is_null_in_the_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = """
    @claim("huge")
    def huge():
        return 10**400


    @claim("small", kind="seeded")
    def small(*, x=1.5):
        return x
    """
    readme = (
        "<!-- claim: huge -->1e400<!-- /claim -->\n<!-- claim: small lt -->1e999<!-- /claim -->\n"
    )
    make_project(tmp_path, logging_claims(claims), {"README.md": readme})
    assert run(tmp_path, "lock") == 0
    capsys.readouterr()
    assert run(tmp_path, "check", "--json", str(tmp_path / "report.json")) == 0
    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    report = json.loads(text, parse_constant=no_constant)
    huge, small = report["occurrences"]
    assert (huge["result"], huge["compared"]) == ("pass", None)
    assert (small["verdict"], small["compared"], small["margin"]) == ("supported", 1.5, None)
    assert "README.md:2 small lt 1e999 value=1.5 pass quantity=value compared=1.5 margin=- " in (
        capsys.readouterr().out
    )
