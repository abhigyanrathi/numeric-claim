import os
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import counting_claims, make_project, write
from numeric_claim.cli import main

SOURCE = Path(__file__).resolve().parents[1] / "src"


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SOURCE), environment.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-m", "numeric_claim", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )


def test_module_help_runs_in_a_subprocess() -> None:
    completed = run_module("--help")
    assert completed.returncode == 0
    assert "usage: numeric-claim" in completed.stdout
    for word in ("lock", "check", "list", "--config", "--root", "--lock", "--json", "--recompute"):
        assert word in completed.stdout


def test_module_exit_statuses_are_those_of_the_report(tmp_path: Path) -> None:
    make_project(
        tmp_path,
        counting_claims(),
        {"value.txt": "42", "README.md": "<!-- claim: answer -->41<!-- /claim -->\n"},
    )
    config = str(tmp_path / "numeric-claim.toml")
    assert run_module("lock", "--config", config).returncode == 0
    failing = run_module("check", "--config", config)
    assert failing.returncode == 1
    assert "README.md:1 answer eq 41 value=42 fail" in failing.stdout
    write(tmp_path, "README.md", "<!-- claim: answer -->42<!-- /claim -->\n")
    assert run_module("check", "--config", config).returncode == 0
    missing = run_module("check", "--config", str(tmp_path / "absent.toml"))
    assert missing.returncode == 2
    assert "numeric-claim: error:" in missing.stderr


def test_main_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "numeric-claim" in capsys.readouterr().out


def test_characters_a_narrow_console_cannot_show_are_escaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_project(
        tmp_path,
        counting_claims(),
        {
            "value.txt": "42",
            "README.md": "<!-- claim: answer -->\N{MINUS SIGN}7<!-- /claim -->\n",
        },
    )
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    config = str(tmp_path / "numeric-claim.toml")
    assert run_module("lock", "--config", config).returncode == 0
    completed = run_module("check", "--config", config)
    assert completed.returncode == 1
    assert "README.md:1 answer eq \\u22127 value=42 fail" in completed.stdout
