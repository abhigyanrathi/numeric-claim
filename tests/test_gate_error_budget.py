"""Gate 2: the precision bounds in cosfin's replications.py, judged by error budgets."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

COMMIT = "39dee29614080f364f8c23c50f2b37279214f1ea"
REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "examples" / "cosfin-39dee29" / "numeric-claim.toml"
SOURCE = "src/cosfin/replications.py"
TABLE_6_7 = {"n": [1024, 2048], "l_trunc": [11.0, 12.0]}
TABLE_6_8 = {"n": [8192, 16384], "l_trunc": [11.0, 12.0]}
LADDERS = {"t67_y05": TABLE_6_7, "t67_y15": TABLE_6_7, "t68_t01": TABLE_6_8, "t68_t1": TABLE_6_8}
LABELS = {"t67_y05": "l_trunc", "t67_y15": "rounding", "t68_t01": "n", "t68_t1": "reference"}
BOUNDS = [
    (SOURCE, 97, "t67_y05", "4e-8"),
    (SOURCE, 97, "t67_y15", "4e-8"),
    (SOURCE, 100, "t68_t01", "1e-8"),
    (SOURCE, 100, "t68_t1", "1e-8"),
]


@pytest.fixture
def checkout(request: pytest.FixtureRequest) -> Path:
    option = request.config.getoption("--cosfin-39dee29")
    if option is None:
        pytest.skip("pass --cosfin-39dee29 PATH to run the error-budget gate")
    return Path(option)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPOSITORY / "src"), environment.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-m", "numeric_claim", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )


def package_files(top: Path) -> dict[str, bytes]:
    return {
        path.relative_to(top).as_posix(): path.read_bytes().replace(b"\r\n", b"\n")
        for path in top.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(top).parts
    }


def assert_installed_package_is(pinned: Path) -> None:
    spec = importlib.util.find_spec("cosfin")
    assert spec is not None
    assert spec.submodule_search_locations
    installed = package_files(Path(spec.submodule_search_locations[0]))
    expected = package_files(pinned)
    assert sorted(installed) == sorted(expected)
    assert [name for name in sorted(expected) if installed[name] != expected[name]] == []


def assert_expected_report(report: dict[str, Any]) -> None:
    claims = {claim["name"]: claim for claim in report["claims"]}
    assert {name: claim["status"] for name, claim in claims.items()} == dict.fromkeys(LABELS, "ok")
    assert {name: claim["budget"]["attribution"] for name, claim in claims.items()} == LABELS
    occurrences = report["occurrences"]
    assert all((o["relation"], o["quantity"]) == ("lt", "error") for o in occurrences)
    found = [(Path(o["file"]).as_posix(), o["line"], o["claim"], o["literal"]) for o in occurrences]
    assert sorted(found) == BOUNDS
    errors = {o["claim"]: o["compared"] for o in occurrences}
    assert {o["claim"]: o["verdict"] for o in occurrences} == {
        "t67_y05": "supported",
        "t67_y15": "rounding-limited" if errors["t67_y15"] < 4e-8 else "violated",
        "t68_t01": "supported",
        "t68_t1": "supported",
    }


def test_the_precision_bounds_are_judged_by_their_error_budgets(
    checkout: Path, tmp_path: Path
) -> None:
    head = (checkout / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    assert head == COMMIT
    assert_installed_package_is(checkout / "src" / "cosfin")

    lock = tmp_path / "lock"
    options = ["--config", str(CONFIG), "--root", str(checkout), "--lock", str(lock)]

    locked = run_cli("lock", *options)
    assert locked.returncode == 0, locked.stdout + locked.stderr
    stored = json.loads(lock.read_text(encoding="utf-8"))["claims"]
    assert sorted(stored) == sorted(LADDERS)
    for name, entry in stored.items():
        budget = entry["budget"]
        assert budget["nudge"] is not None, name
        assert budget["jitter"] is not None, name
        assert {
            parameter: [rung["setting"] for rung in rungs]
            for parameter, rungs in budget["ladders"].items()
        } == LADDERS[name]

    checked = run_cli("check", *options, "--json", str(tmp_path / "check.json"))
    print(checked.stdout)
    assert checked.returncode == 1, checked.stdout + checked.stderr
    assert_expected_report(json.loads((tmp_path / "check.json").read_text(encoding="utf-8")))

    again = run_cli("check", "--recompute", *options, "--json", str(tmp_path / "recompute.json"))
    print(again.stdout)
    assert again.returncode == 1, again.stdout + again.stderr
    assert_expected_report(json.loads((tmp_path / "recompute.json").read_text(encoding="utf-8")))

    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert status.stdout == ""
