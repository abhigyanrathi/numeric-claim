import importlib.metadata
import platform
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from helpers import write
from numeric_claim.claim import ClaimContext, ClaimSpec
from numeric_claim.errors import ClaimError
from numeric_claim.fingerprint import compute

SOURCE = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'


def digest(
    root: Path,
    *,
    paths: tuple[str, ...] = ("src",),
    packages: tuple[str, ...] = (),
    distributions: tuple[str, ...] = (),
) -> str:
    spec = ClaimSpec("c", "exact", 0.0, 0.0, packages, paths)
    context = ClaimContext(root, root)
    return compute(spec, root / "claims.py", context, distributions).digest


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write(tmp_path, "claims.py", "VALUE = 1\n")
    write(tmp_path, "src/mod.py", SOURCE)
    write(tmp_path, "src/data.csv", "a,b\n1,2\n")
    return tmp_path


def fake_version(version: str) -> Callable[[str], str]:
    return lambda name: version


def test_the_digest_is_a_sha256_hex_string_and_repeatable(project: Path) -> None:
    first = digest(project)
    assert len(first) == 64
    assert int(first, 16) >= 0
    assert digest(project) == first


def test_comments_blank_lines_and_formatting_do_not_change_it(project: Path) -> None:
    before = digest(project)
    reformatted = (
        "# a leading comment\n\n\n"
        "def add( a,b ):\n"
        '    """Add two numbers."""\n\n'
        "    return (a+b)  # the sum\n"
    )
    write(project, "src/mod.py", reformatted)
    assert digest(project) == before


def test_a_code_edit_changes_it(project: Path) -> None:
    before = digest(project)
    write(project, "src/mod.py", SOURCE.replace("a + b", "a - b"))
    assert digest(project) != before


def test_a_docstring_edit_changes_it(project: Path) -> None:
    before = digest(project)
    write(project, "src/mod.py", SOURCE.replace("Add two numbers.", "Add them."))
    assert digest(project) != before


def test_a_non_python_file_in_a_declared_path_changes_it(project: Path) -> None:
    before = digest(project)
    write(project, "src/data.csv", "a,b\n1,3\n")
    assert digest(project) != before


def test_a_new_file_in_a_declared_path_changes_it(project: Path) -> None:
    before = digest(project)
    write(project, "src/extra.txt", "new\n")
    assert digest(project) != before


def test_files_outside_declared_paths_do_not_change_it(project: Path) -> None:
    before = digest(project)
    write(project, "elsewhere/mod.py", "x = 1\n")
    write(project, "README.md", "hello\n")
    assert digest(project) == before


def test_bytecode_caches_are_ignored(project: Path) -> None:
    before = digest(project)
    write(project, "src/__pycache__/mod.cpython-312.pyc", "junk")
    write(project, "src/stale.pyc", "junk")
    assert digest(project) == before


def test_a_python_file_that_does_not_parse_is_hashed_by_its_bytes(project: Path) -> None:
    write(project, "src/broken.py", "def broken(:\n")
    before = digest(project)
    write(project, "src/broken.py", "def broken(:  \n")
    assert digest(project) != before


def test_the_defining_module_is_covered(project: Path) -> None:
    before = digest(project)
    write(project, "claims.py", "VALUE = 2\n")
    assert digest(project) != before


def test_a_distribution_version_change_changes_it(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", fake_version("1.0"))
    first = digest(project, distributions=("anything",))
    monkeypatch.setattr(importlib.metadata, "version", fake_version("2.0"))
    second = digest(project, distributions=("anything",))
    assert first != second


def test_the_environment_block_records_interpreter_platform_and_versions(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", fake_version("4.5.6"))
    spec = ClaimSpec("c", "exact", 0.0, 0.0, (), ("src",))
    context = ClaimContext(project, project)
    environment = compute(spec, project / "claims.py", context, ("lib",)).environment
    assert environment == {
        "distributions": {"lib": "4.5.6"},
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_platform": sys.platform,
    }


def test_an_absent_distribution_is_recorded_as_not_installed(project: Path) -> None:
    spec = ClaimSpec("c", "exact", 0.0, 0.0, (), ())
    context = ClaimContext(project, project)
    name = "numeric-claim-no-such-distribution"
    environment = compute(spec, project / "claims.py", context, (name,)).environment
    assert environment["distributions"] == {name: "not installed"}


PATCHES: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "python_version": lambda mp: mp.setattr(platform, "python_version", lambda: "9.9.9"),
    "python_implementation": lambda mp: mp.setattr(platform, "python_implementation", lambda: "X"),
    "machine": lambda mp: mp.setattr(platform, "machine", lambda: "other"),
    "sys_platform": lambda mp: mp.setattr(sys, "platform", "plan9"),
}


@pytest.mark.parametrize("name", sorted(PATCHES))
def test_interpreter_and_platform_are_part_of_it(
    project: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    before = digest(project)
    PATCHES[name](monkeypatch)
    assert digest(project) != before


def test_a_missing_declared_path_is_an_error(project: Path) -> None:
    with pytest.raises(ClaimError, match="'nowhere' does not exist"):
        digest(project, paths=("src", "nowhere"))


def test_a_single_file_path_is_covered(project: Path) -> None:
    before = digest(project, paths=("src/data.csv",))
    write(project, "src/mod.py", SOURCE.replace("+", "-"))
    assert digest(project, paths=("src/data.csv",)) == before
    write(project, "src/data.csv", "changed\n")
    assert digest(project, paths=("src/data.csv",)) != before


def test_a_declared_package_is_covered(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(project, "nc_fp_pkg/__init__.py", "from .core import value\n")
    write(project, "nc_fp_pkg/core.py", "def value():\n    return 1\n")
    monkeypatch.syspath_prepend(str(project))
    before = digest(project, paths=(), packages=("nc_fp_pkg",))
    assert digest(project, paths=(), packages=("nc_fp_pkg",)) == before
    write(project, "nc_fp_pkg/core.py", "def value():\n    return 2\n")
    assert digest(project, paths=(), packages=("nc_fp_pkg",)) != before


def test_a_single_module_package_is_covered(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(project, "nc_fp_single.py", "VALUE = 1\n")
    monkeypatch.syspath_prepend(str(project))
    before = digest(project, paths=(), packages=("nc_fp_single",))
    write(project, "nc_fp_single.py", "VALUE = 2\n")
    assert digest(project, paths=(), packages=("nc_fp_single",)) != before


def test_an_unknown_package_is_an_error(project: Path) -> None:
    with pytest.raises(ClaimError, match="not found"):
        digest(project, paths=(), packages=("nc_fp_no_such_package",))
