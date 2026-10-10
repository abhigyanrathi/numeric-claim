"""Gate 1: the test count in a cosfin README, bound by pattern to a computed claim."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

COMMIT = "5f51ee5135fd4cbd743e24d07d92969dde6fc4d8"
REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "examples" / "cosfin-5f51ee5" / "numeric-claim.toml"


@pytest.fixture
def checkout(request: pytest.FixtureRequest) -> Path:
    option = request.config.getoption("--cosfin-5f51ee5")
    if option is None:
        pytest.skip("pass --cosfin-5f51ee5 PATH to run the claim-binding gate")
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
    assert claims["test_count"]["status"] == "ok"
    assert claims["test_count"]["value"] == 264
    assert [
        (o["file"], o["line"], o["claim"], o["literal"], o["result"]) for o in report["occurrences"]
    ] == [
        ("README.md", 32, "test_count", "264", "pass"),
        ("README.md", 94, "test_count", "249", "fail"),
    ]


def test_the_stale_test_count_is_caught(checkout: Path, tmp_path: Path) -> None:
    head = (checkout / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    assert head == COMMIT

    options = ["--config", str(CONFIG), "--root", str(checkout), "--lock", str(tmp_path / "lock")]

    locked = run_cli("lock", *options)
    assert locked.returncode == 0, locked.stdout + locked.stderr
    stored = json.loads((tmp_path / "lock").read_text(encoding="utf-8"))
    assert stored["claims"]["test_count"]["value"] == 264

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
