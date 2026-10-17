import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import pytest
from scipy import stats

from helpers import DEFAULT_CONFIG, json_report, make_project, run, write
from numeric_claim import lockfile
from numeric_claim.budget import Budget
from numeric_claim.cli import main
from numeric_claim.lockfile import LockEntry
from numeric_claim.montecarlo import Summary

IDENTITY = {
    "GIT_AUTHOR_NAME": "numeric-claim tests",
    "GIT_AUTHOR_EMAIL": "tests@example.invalid",
    "GIT_COMMITTER_NAME": "numeric-claim tests",
    "GIT_COMMITTER_EMAIL": "tests@example.invalid",
}
FINGERPRINT = "a" * 64
ENVIRONMENT = {"machine": "x86_64", "numeric_claim": "0.1.0.dev0", "python_version": "3.12.3"}
BROKEN = "raise RuntimeError('diff must not load the claims module')\n"


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **IDENTITY},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def exact(value: int, fingerprint: str = FINGERPRINT, **environment: str) -> LockEntry:
    return LockEntry("exact", value, fingerprint, {**ENVIRONMENT, **environment})


def seeded(
    value: float, *, rtol: float = 1e-12, atol: float = 0.0, sigma: float | None = None
) -> LockEntry:
    budget = None if sigma is None else Budget(sigma, None, {})
    return LockEntry("seeded", value, FINGERPRINT, ENVIRONMENT, rtol, atol, budget)


def randomized(mean: float, sd: float = 1.0, replicates: int = 32) -> LockEntry:
    summary = Summary(mean, sd, replicates)
    return LockEntry("random", mean, FINGERPRINT, ENVIRONMENT, 0.0, 0.0, None, summary)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "--quiet")
    make_project(tmp_path, BROKEN)
    return tmp_path


def commit(root: Path, entries: Mapping[str, LockEntry], message: str = "Lock") -> None:
    lockfile.write(root / "claims.lock.json", entries)
    git(root, "add", "--", "numeric-claim.toml", "claims.py", "claims.lock.json")
    git(root, "commit", "--quiet", "-m", message)


def diff(
    root: Path, current: Mapping[str, LockEntry] | None, *revision: str
) -> tuple[int, dict[str, Any]]:
    if current is not None:
        lockfile.write(root / "claims.lock.json", current)
    return json_report(root, "diff", *revision)


def statuses(report: dict[str, Any]) -> list[tuple[str, str]]:
    return [(change["name"], change["status"]) for change in report["claims"]]


def test_an_unchanged_lock_has_no_differences(repository: Path) -> None:
    commit(repository, {"count": exact(264), "pi": seeded(3.14159), "mc": randomized(1.0)})
    status, report = diff(repository, None)
    assert status == 0
    assert statuses(report) == [("count", "unchanged"), ("mc", "unchanged"), ("pi", "unchanged")]


def test_an_exact_claim_moves_when_its_value_changes(repository: Path) -> None:
    commit(repository, {"count": exact(264)})
    status, report = diff(repository, {"count": exact(265)})
    assert status == 1
    assert report["claims"] == [
        {
            "name": "count",
            "status": "moved",
            "old_value": 264,
            "new_value": 265,
            "fingerprint_changed": False,
            "environment": [],
        }
    ]


def test_a_changed_kind_is_moved_even_with_an_equal_value(repository: Path) -> None:
    commit(repository, {"x": exact(2)})
    status, report = diff(repository, {"x": seeded(2.0)})
    assert (status, statuses(report)) == (1, [("x", "moved")])


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (seeded(1.0, rtol=1e-3), seeded(1.0009, rtol=1e-9), "within-noise"),
        (seeded(1.0, rtol=1e-3), seeded(1.0011, rtol=1.0), "moved"),
        (seeded(1.0, rtol=1e-3), seeded(0.9989), "moved"),
        (seeded(0.0, atol=0.5), seeded(0.4), "within-noise"),
        (seeded(0.0, atol=0.5), seeded(-0.6), "moved"),
    ],
)
def test_a_seeded_claim_moves_beyond_its_old_tolerance(
    repository: Path, old: LockEntry, new: LockEntry, expected: str
) -> None:
    commit(repository, {"s": old})
    status, report = diff(repository, {"s": new})
    assert statuses(report) == [("s", expected)]
    assert status == (1 if expected == "moved" else 0)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (seeded(1.0, sigma=0.1), seeded(1.29, sigma=0.01), "within-noise"),
        (seeded(1.0, sigma=0.1), seeded(1.31, sigma=0.01), "moved"),
        (seeded(1.0, sigma=0.01), seeded(1.31, sigma=0.2), "within-noise"),
        (seeded(1.0, sigma=0.1), seeded(1.29), "within-noise"),
        (seeded(1.0), seeded(1.29, sigma=0.1), "within-noise"),
        (seeded(1.0, sigma=0.1), seeded(0.69), "moved"),
    ],
)
def test_a_budgeted_claim_moves_beyond_three_sigma_of_either_entry(
    repository: Path, old: LockEntry, new: LockEntry, expected: str
) -> None:
    commit(repository, {"b": old})
    status, report = diff(repository, {"b": new})
    assert statuses(report) == [("b", expected)]
    assert status == (1 if expected == "moved" else 0)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (randomized(1.0), randomized(1.05), "within-noise"),
        (randomized(1.0), randomized(1.0, sd=2.0, replicates=64), "unchanged"),
        (randomized(1.0), randomized(2.0), "moved"),
        (randomized(1.0), randomized(0.0, sd=3.0, replicates=8), "within-noise"),
        (randomized(1.0, sd=0.0), randomized(1.0 + 1e-12, sd=0.0), "moved"),
    ],
)
def test_a_random_claim_moves_when_welchs_test_separates_the_means(
    repository: Path, old: LockEntry, new: LockEntry, expected: str
) -> None:
    commit(repository, {"r": old})
    status, report = diff(repository, {"r": new})
    assert statuses(report) == [("r", expected)]
    assert status == (1 if expected == "moved" else 0)


def test_alpha_comes_from_the_configuration(repository: Path) -> None:
    shift = 0.6
    p = stats.ttest_ind_from_stats(1.0, 1.0, 32, 1.0 + shift, 1.0, 32, equal_var=False).pvalue
    assert 0.01 < p < 0.05
    commit(repository, {"r": randomized(1.0)})
    status, report = diff(repository, {"r": randomized(1.0 + shift)})
    assert (status, statuses(report)) == (0, [("r", "within-noise")])
    write(repository, "numeric-claim.toml", DEFAULT_CONFIG + "alpha = 0.05\n")
    status, report = diff(repository, None)
    assert (status, statuses(report)) == (1, [("r", "moved")])


def test_a_claim_added_and_one_removed(repository: Path) -> None:
    commit(repository, {"gone": exact(1), "kept": exact(2)})
    status, report = diff(repository, {"kept": exact(2), "new": seeded(0.5)})
    assert status == 1
    assert report["claims"] == [
        {
            "name": "gone",
            "status": "removed",
            "old_value": 1,
            "new_value": None,
            "fingerprint_changed": None,
            "environment": None,
        },
        {
            "name": "kept",
            "status": "unchanged",
            "old_value": 2,
            "new_value": 2,
            "fingerprint_changed": False,
            "environment": [],
        },
        {
            "name": "new",
            "status": "added",
            "old_value": None,
            "new_value": 0.5,
            "fingerprint_changed": None,
            "environment": None,
        },
    ]


def test_without_a_lock_file_at_the_revision_every_claim_is_added(repository: Path) -> None:
    git(repository, "add", "--", "numeric-claim.toml", "claims.py")
    git(repository, "commit", "--quiet", "-m", "Configure")
    status, report = diff(repository, {"a": exact(1), "b": seeded(2.0)})
    assert (status, statuses(report)) == (1, [("a", "added"), ("b", "added")])


def test_any_revision_may_be_compared(repository: Path) -> None:
    commit(repository, {"count": exact(1)}, "First")
    commit(repository, {"count": exact(2)}, "Second")
    status, report = diff(repository, None)
    assert (status, statuses(report), report["revision"]) == (0, [("count", "unchanged")], "HEAD")
    status, report = diff(repository, None, "HEAD~1")
    assert (status, report["revision"]) == (1, "HEAD~1")
    assert report["claims"][0] == {
        "name": "count",
        "status": "moved",
        "old_value": 1,
        "new_value": 2,
        "fingerprint_changed": False,
        "environment": [],
    }
    first = git(repository, "rev-list", "--max-parents=0", "HEAD").strip()
    status, report = diff(repository, None, first)
    assert (status, report["claims"][0]["old_value"]) == (1, 1)


def test_an_unknown_revision_is_an_error_quoting_git(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(repository, {"count": exact(1)})
    assert run(repository, "diff", "no-such-revision") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "numeric-claim: error: git show no-such-revision:./claims.lock.json failed: fatal: "
    )
    assert "no-such-revision" in captured.err.split("failed: ", 1)[1]


def test_a_directory_that_is_not_a_repository_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    project = tmp_path / "plain"
    make_project(project, BROKEN)
    lockfile.write(project / "claims.lock.json", {"count": exact(1)})
    assert run(project, "diff") == 2
    error = capsys.readouterr().err
    assert error.startswith("numeric-claim: error: git show HEAD:./claims.lock.json failed: ")
    assert "not a git repository" in error


def test_a_missing_git_is_an_error(
    repository: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(repository, {"count": exact(1)})
    monkeypatch.setenv("PATH", str(repository / "nowhere"))
    assert run(repository, "diff") == 2
    assert "numeric-claim: error: cannot run git" in capsys.readouterr().err


def test_a_missing_current_lock_file_is_a_configuration_error(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    git(repository, "add", "--", "numeric-claim.toml", "claims.py")
    git(repository, "commit", "--quiet", "-m", "Configure")
    assert run(repository, "diff") == 2
    assert "claims.lock.json does not exist; run `numeric-claim lock`" in capsys.readouterr().err


def test_an_invalid_lock_file_at_the_revision_is_a_configuration_error(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(repository, "claims.lock.json", "not json\n")
    git(repository, "add", "--", "numeric-claim.toml", "claims.py", "claims.lock.json")
    git(repository, "commit", "--quiet", "-m", "Break")
    lockfile.write(repository / "claims.lock.json", {"count": exact(1)})
    assert run(repository, "diff") == 2
    assert "invalid lock file HEAD:./claims.lock.json" in capsys.readouterr().err


def test_a_revision_applies_to_diff_only_and_recompute_does_not(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for command in ("lock", "check", "list"):
        assert run(repository, command, "HEAD") == 2
        assert "a revision applies to the diff command only" in capsys.readouterr().err
    assert run(repository, "diff", "--recompute") == 2
    assert "--recompute applies to the check command only" in capsys.readouterr().err


def test_the_reports_show_values_fingerprints_and_environment_changes(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(repository, {"count": exact(264), "pi": seeded(3.14159)})
    capsys.readouterr()
    rebuilt = exact(264, "b" * 64, numeric_claim="0.2.0", python_version="3.13.0")
    status, report = diff(repository, {"count": rebuilt, "pi": seeded(3.2), "new": exact(1)})
    assert status == 1
    assert report == {
        "command": "diff",
        "exit_code": 1,
        "revision": "HEAD",
        "claims": [
            {
                "name": "count",
                "status": "unchanged",
                "old_value": 264,
                "new_value": 264,
                "fingerprint_changed": True,
                "environment": ["numeric_claim", "python_version"],
            },
            {
                "name": "new",
                "status": "added",
                "old_value": None,
                "new_value": 1,
                "fingerprint_changed": None,
                "environment": None,
            },
            {
                "name": "pi",
                "status": "moved",
                "old_value": 3.14159,
                "new_value": 3.2,
                "fingerprint_changed": False,
                "environment": [],
            },
        ],
    }
    assert capsys.readouterr().out.splitlines() == [
        "claim count unchanged old=264 new=264 fingerprint=changed "
        "environment=numeric_claim,python_version",
        "claim new added old=- new=1",
        "claim pi moved old=3.14159 new=3.2 fingerprint=same environment=-",
        "3 claims against HEAD (1 added, 1 moved, 1 unchanged): exit 1",
    ]


MOVED = [
    "claim count moved old=1 new=2 fingerprint=same environment=-",
    "1 claims against HEAD~1 (1 moved): exit 1",
]


@pytest.fixture
def moved(repository: Path) -> Path:
    commit(repository, {"count": exact(1)}, "First")
    commit(repository, {"count": exact(2)}, "Second")
    return repository


def options(root: Path) -> dict[str, str]:
    return {
        "--config": str(root / "numeric-claim.toml"),
        "--root": str(root),
        "--lock": str(root / "claims.lock.json"),
        "--json": str(root / "report.json"),
    }


def arguments(given: Mapping[str, str]) -> list[str]:
    return [text for option in given.items() for text in option]


def invoke(
    root: Path, capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, list[str], dict[str, Any]]:
    report = root / "report.json"
    report.unlink(missing_ok=True)
    status = main(list(argv))
    lines = capsys.readouterr().out.splitlines()
    return status, lines, json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.parametrize("option", ["--config", "--root", "--lock", "--json"])
def test_the_revision_may_follow_an_option(
    moved: Path, capsys: pytest.CaptureFixture[str], option: str
) -> None:
    rest = options(moved)
    between = [option, rest.pop(option)]
    expected = invoke(moved, capsys, "diff", "HEAD~1", *between, *arguments(rest))
    assert expected[:2] == (1, MOVED)
    assert invoke(moved, capsys, "diff", *between, "HEAD~1", *arguments(rest)) == expected


def test_options_may_come_before_the_command_and_the_revision_last(
    moved: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    given = arguments(options(moved))
    expected = invoke(moved, capsys, "diff", "HEAD~1", *given)
    assert expected[:2] == (1, MOVED)
    assert invoke(moved, capsys, *given, "diff", "HEAD~1") == expected
    assert invoke(moved, capsys, "diff", *given, "HEAD~1") == expected


def test_a_revision_starting_with_a_dash_is_refused_before_git_runs(
    moved: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail(f"diff ran {args!r}")

    monkeypatch.setattr("numeric_claim.diff.subprocess.run", forbidden)
    config = ["--config", str(moved / "numeric-claim.toml")]
    for argv in ([*config, "diff", "--", "--output=x"], ["diff", *config, "--", "--output=x"]):
        assert main(argv) == 2
        assert capsys.readouterr().err == "numeric-claim: error: invalid revision '--output=x'\n"


def test_a_revision_after_an_option_still_applies_to_diff_only(
    moved: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = str(moved / "numeric-claim.toml")
    for command in ("lock", "check", "list"):
        assert main([command, "--config", config, "HEAD~1"]) == 2
        assert "a revision applies to the diff command only" in capsys.readouterr().err
