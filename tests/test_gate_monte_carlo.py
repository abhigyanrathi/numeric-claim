"""Gate 3: cosfin's delta-hedging numbers, read as statements about expected values."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from helpers import assert_installed_package_is

COMMIT = "39dee29614080f364f8c23c50f2b37279214f1ea"
REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "examples" / "cosfin-39dee29-hedging" / "numeric-claim.toml"
README = "README.md"
TESTS = "tests/test_ch5_hedging.py"
CLAIMS = ["merton_gbm_208", "merton_mean_208", "ratio_13_208", "ratio_13_52"]
OCCURRENCES = [
    (README, 117, "ratio_13_208", "eq", "3.82", "violated"),
    (README, 118, "ratio_13_52", "eq", "1.93", "violated"),
    (README, 118, "merton_gbm_208", "eq", "8.8", "violated"),
    (README, 119, "merton_mean_208", "eq", "\N{MINUS SIGN}2.46", "violated"),
    (TESTS, 25, "ratio_13_208", "gt", "3.0", "supported"),
    (TESTS, 25, "ratio_13_208", "lt", "5.0", "supported"),
    (TESTS, 26, "ratio_13_52", "gt", "1.6", "supported"),
    (TESTS, 26, "ratio_13_52", "lt", "2.4", "supported"),
    (TESTS, 39, "merton_gbm_208", "gt", "4.0", "supported"),
    (TESTS, 42, "merton_mean_208", "lt", "-1.0", "supported"),
]


@pytest.fixture
def checkout(request: pytest.FixtureRequest) -> Path:
    option = request.config.getoption("--cosfin-39dee29")
    if option is None:
        pytest.skip("pass --cosfin-39dee29 PATH to run the Monte Carlo gate")
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


def assert_expected_report(report: dict[str, Any]) -> None:
    claims = {claim["name"]: claim for claim in report["claims"]}
    assert {name: claim["status"] for name, claim in claims.items()} == dict.fromkeys(CLAIMS, "ok")
    assert all(claim["random"]["replicates"] == 64 for claim in claims.values())
    found = [
        (
            Path(o["file"]).as_posix(),
            o["line"],
            o["claim"],
            o["relation"],
            o["literal"],
            o["verdict"],
        )
        for o in report["occurrences"]
    ]
    assert sorted(found) == sorted(OCCURRENCES)


def test_the_hedging_numbers_are_judged_as_statements_about_expected_values(
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
    assert sorted(stored) == CLAIMS
    assert all((entry["kind"], entry["replicates"]) == ("random", 64) for entry in stored.values())

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
