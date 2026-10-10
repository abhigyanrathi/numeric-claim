import sys
from pathlib import Path

import pytest

from helpers import make_project, run, write
from numeric_claim.cli import main
from numeric_claim.config import DEFAULT_DISTRIBUTIONS, DEFAULT_LOCK, DEFAULT_SCAN, load_config

HEADER = 'claims = ["claims.py"]\n'
BIND = "[[bind]]\nfiles = ['README.md']\npattern = '(\\d+) tests'\nclaim = 'x'\n"
SIMPLE_CLAIM = "from numeric_claim import claim\n\n\n@claim('x')\ndef x():\n    return 1\n"


def error_of(root: Path, capsys: pytest.CaptureFixture[str]) -> str:
    assert run(root, "check") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (HEADER + "bogus = 1\n", "unknown key(s): bogus"),
        ("distributions = []\n", "missing required key 'claims'"),
        ("claims = []\n", "'claims' must not be empty"),
        ('claims = "claims.py"\n', "'claims' must be a list of strings"),
        (HEADER + "lock = 3\n", "'lock' must be a string"),
        (HEADER + 'scan = "*.md"\n', "'scan' must be a list of strings"),
        (HEADER + 'bind = "x"\n', "'bind' must be an array of tables"),
        ("claims = [\n", "invalid TOML"),
        (HEADER + BIND + "extra = 1\n", "[[bind]] entry 1: unknown key(s): extra"),
        (HEADER + "[[bind]]\nfiles = ['README.md']\nclaim = 'x'\n", "'pattern' is required"),
        (
            HEADER + "[[bind]]\nfiles = ['README.md']\npattern = '(a)'\n",
            "missing required key 'claim'",
        ),
        (HEADER + "[[bind]]\npattern = '(a)'\nclaim = 'x'\n", "missing required key 'files'"),
        (
            HEADER + "[[bind]]\nfiles = ['README.md']\npattern = '\\d+ tests'\nclaim = 'x'\n",
            "exactly one capture group, found 0",
        ),
        (
            HEADER + "[[bind]]\nfiles = ['README.md']\npattern = '(\\d+) (tests)'\nclaim = 'x'\n",
            "exactly one capture group, found 2",
        ),
        (
            HEADER + "[[bind]]\nfiles = ['README.md']\npattern = '(\\d+'\nclaim = 'x'\n",
            "invalid pattern",
        ),
        (HEADER + BIND + 'relation = "ne"\n', "unknown relation 'ne'"),
        (
            HEADER + "[[bind]]\nfiles = ['MISSING.md']\npattern = '(a)'\nclaim = 'x'\n",
            "'MISSING.md' matches no file",
        ),
        (
            HEADER + "[[bind]]\nfiles = ['README.md', 'docs/*.md']\npattern = '(a)'\nclaim = 'x'\n",
            "'docs/*.md' matches no file",
        ),
    ],
)
def test_configuration_errors_exit_with_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], config: str, message: str
) -> None:
    make_project(tmp_path, SIMPLE_CLAIM, {"README.md": "12 tests\n"}, config)
    assert message in error_of(tmp_path, capsys)


def test_a_missing_configuration_file_exits_with_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path, "check") == 2
    assert "cannot read configuration" in capsys.readouterr().err


def test_a_root_that_is_not_a_directory_exits_with_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_project(tmp_path, SIMPLE_CLAIM)
    arguments = ["check", "--config", str(tmp_path / "numeric-claim.toml")]
    assert main([*arguments, "--root", str(tmp_path / "nowhere")]) == 2
    assert "is not a directory" in capsys.readouterr().err


def test_the_same_claim_name_twice_in_one_module_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('same')\ndef first():\n    return 1\n\n\n"
        "@claim('same')\ndef second():\n    return 2\n"
    )
    make_project(tmp_path, claims)
    assert "claim 'same' is defined twice" in error_of(tmp_path, capsys)


def test_the_same_claim_name_in_two_modules_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = 'claims = ["claims.py", "other.py"]\n'
    make_project(tmp_path, SIMPLE_CLAIM, {"other.py": SIMPLE_CLAIM}, config)
    assert "claim 'x' is defined twice" in error_of(tmp_path, capsys)


def test_a_claim_imported_into_a_module_is_not_defined_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path, "nc_shared_claims.py", SIMPLE_CLAIM)
    make_project(tmp_path, "from nc_shared_claims import x\n")
    try:
        assert run(tmp_path, "list") == 0
    finally:
        sys.modules.pop("nc_shared_claims", None)
    assert "0 claims" in capsys.readouterr().out


def test_a_missing_claims_module_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_project(tmp_path, "", config='claims = ["absent.py"]\n')
    assert "claims module not found" in error_of(tmp_path, capsys)


@pytest.mark.parametrize(
    "source",
    [
        "raise RuntimeError('broken import')\n",
        "def broken(:\n",
        "from numeric_claim import claim\n\n@claim('x', kind='bogus')\ndef x():\n    return 1\n",
        "import sys\nsys.exit(0)\n",
    ],
)
def test_a_claims_module_that_fails_to_load_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    make_project(tmp_path, source)
    assert "error while loading claims module" in error_of(tmp_path, capsys)


def test_an_unknown_command_and_a_missing_command_are_usage_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["bogus"]) == 2
    assert main([]) == 2
    assert "usage: numeric-claim" in capsys.readouterr().err


def test_recompute_applies_to_check_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_project(tmp_path, SIMPLE_CLAIM)
    for command in ("lock", "list"):
        assert run(tmp_path, command, "--recompute") == 2
        assert "--recompute applies to the check command only" in capsys.readouterr().err
    assert not (tmp_path / DEFAULT_LOCK).exists()


def test_defaults_are_applied(tmp_path: Path) -> None:
    write(tmp_path, "numeric-claim.toml", HEADER)
    config = load_config(tmp_path / "numeric-claim.toml")
    assert config.lock == tmp_path.resolve() / "claims.lock.json"
    assert config.distributions == DEFAULT_DISTRIBUTIONS == ("numpy", "scipy", "mpmath")
    assert config.scan == DEFAULT_SCAN == ("**/*.md", "**/*.tex")
    assert config.root == tmp_path.resolve()
    assert config.bindings == ()
