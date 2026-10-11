import json
from pathlib import Path
from typing import Any

import pytest

from helpers import DEFAULT_CONFIG, calls, counting_claims, make_project, run, write
from numeric_claim.cli import main

ANSWER_42 = "Answer: <!-- claim: answer -->42<!-- /claim -->\n"


def report_of(root: Path, *args: str) -> tuple[int, dict[str, Any]]:
    path = root / "report.json"
    status = run(root, *args, "--json", str(path))
    return status, json.loads(path.read_text(encoding="utf-8"))


def statuses(report: dict[str, Any]) -> dict[str, str]:
    return {claim["name"]: claim["status"] for claim in report["claims"]}


def results(report: dict[str, Any]) -> list[tuple[int, str]]:
    return [(item["line"], item["result"]) for item in report["occurrences"]]


@pytest.fixture
def locked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    make_project(tmp_path, counting_claims(), {"value.txt": "42", "README.md": ANSWER_42})
    assert run(tmp_path, "lock") == 0
    capsys.readouterr()
    return tmp_path


def test_a_matching_fingerprint_reuses_the_locked_value_without_calling_the_claim(
    locked: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert calls(locked) == 1
    assert run(locked, "check") == 0
    assert calls(locked) == 1
    output = capsys.readouterr().out
    assert "claim answer [exact] ok value=42 (lock)" in output
    assert "README.md:1 answer eq 42 value=42 pass" in output
    assert output.splitlines()[-1] == "1 claims (1 ok), 1 occurrences (1 pass): exit 0"


def test_recompute_calls_the_claim(locked: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(locked, "check", "--recompute") == 0
    assert calls(locked) == 2
    assert "claim answer [exact] ok value=42 (computed)" in capsys.readouterr().out


def test_check_never_writes_the_lock_file(locked: Path) -> None:
    lock = locked / "claims.lock.json"
    before = lock.read_bytes()
    write(locked, "value.txt", "43")
    assert run(locked, "check", "--recompute") == 1
    assert lock.read_bytes() == before


def test_a_claim_without_a_lock_entry_is_unlocked_and_nothing_is_written(tmp_path: Path) -> None:
    make_project(tmp_path, counting_claims(), {"value.txt": "42", "README.md": ANSWER_42})
    status, report = report_of(tmp_path, "check")
    assert status == 1
    assert statuses(report) == {"answer": "unlocked"}
    assert results(report) == [(1, "pass")]
    assert report["exit_code"] == 1
    assert not (tmp_path / "claims.lock.json").exists()


def test_a_changed_value_under_an_unchanged_fingerprint_is_drift_when_recomputed(
    locked: Path,
) -> None:
    write(locked, "value.txt", "43")
    assert run(locked, "check") == 0
    status, report = report_of(locked, "check", "--recompute")
    assert status == 1
    assert statuses(report) == {"answer": "drift"}
    claim = report["claims"][0]
    assert (claim["value"], claim["locked_value"], claim["source"]) == (43, 42, "computed")
    assert results(report) == [(1, "fail")]


def test_a_changed_fingerprint_with_the_same_value_is_ok_with_a_note(
    locked: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (locked / "claims.py").read_text(encoding="utf-8")
    write(locked, "claims.py", source + "\nUNUSED = 1\n")
    assert run(locked, "check") == 0
    assert calls(locked) == 2
    output = capsys.readouterr().out
    assert "ok value=42 (computed): fingerprint changed; run `numeric-claim lock`" in output


def test_a_changed_fingerprint_with_a_new_value_is_drift(locked: Path) -> None:
    write(locked, "value.txt", "50")
    source = (locked / "claims.py").read_text(encoding="utf-8")
    write(locked, "claims.py", source + "\nUNUSED = 1\n")
    status, report = report_of(locked, "check")
    assert status == 1
    assert statuses(report) == {"answer": "drift"}
    assert results(report) == [(1, "fail")]


@pytest.mark.parametrize(
    ("options", "locked_text", "new_text", "expected"),
    [
        ("", "1.0", "1.0000000000001", "ok"),
        ("", "1.0", "1.000000001", "drift"),
        (", rtol=1e-3", "1.0", "1.0009", "ok"),
        (", rtol=1e-3", "1.0", "1.0011", "drift"),
        (", atol=1e-3", "0.0", "0.0009", "ok"),
        (", atol=1e-3", "0.0", "0.0011", "drift"),
    ],
)
def test_a_seeded_claim_is_compared_within_its_tolerance(
    tmp_path: Path, options: str, locked_text: str, new_text: str, expected: str
) -> None:
    claims = counting_claims(kind="seeded", options=options, cast="float")
    make_project(tmp_path, claims, {"value.txt": locked_text})
    assert run(tmp_path, "lock") == 0
    write(tmp_path, "value.txt", new_text)
    status, report = report_of(tmp_path, "check", "--recompute")
    assert statuses(report) == {"answer": expected}
    assert status == (0 if expected == "ok" else 1)


def test_an_exact_claim_that_moves_by_a_hair_is_drift(tmp_path: Path) -> None:
    make_project(tmp_path, counting_claims(cast="float"), {"value.txt": "1.0"})
    assert run(tmp_path, "lock") == 0
    write(tmp_path, "value.txt", "1.0000000000000002")
    status, report = report_of(tmp_path, "check", "--recompute")
    assert status == 1
    assert statuses(report) == {"answer": "drift"}


def test_a_claim_that_raises_is_an_error_and_its_occurrences_are_claim_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('boom')\ndef boom():\n    raise RuntimeError('boom')\n"
    )
    readme = "<!-- claim: boom -->1<!-- /claim -->\n"
    make_project(tmp_path, claims, {"README.md": readme})
    status, report = report_of(tmp_path, "check")
    assert status == 1
    assert statuses(report) == {"boom": "error"}
    assert report["claims"][0]["note"] == "RuntimeError: boom"
    assert results(report) == [(1, "claim-error")]
    output = capsys.readouterr().out
    assert "claim boom [exact] error value=-: RuntimeError: boom" in output
    assert "README.md:1 boom eq 1 value=- claim-error: RuntimeError: boom" in output


def test_lock_writes_nothing_and_exits_1_when_a_claim_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('fine')\ndef fine():\n    return 1\n\n\n"
        "@claim('boom')\ndef boom():\n    return 'not a number'\n"
    )
    make_project(tmp_path, claims)
    assert run(tmp_path, "lock") == 1
    assert not (tmp_path / "claims.lock.json").exists()
    output = capsys.readouterr().out
    assert "locked fine = 1" in output
    assert "error boom: claim returned str" in output
    assert "nothing written: 1 of 2 claims failed" in output


def test_a_failed_lock_leaves_an_existing_lock_untouched(locked: Path) -> None:
    lock = locked / "claims.lock.json"
    before = lock.read_bytes()
    write(locked, "value.txt", "not a number")
    assert run(locked, "lock") == 1
    assert lock.read_bytes() == before


def test_document_occurrences_are_judged_individually(tmp_path: Path) -> None:
    readme = (
        "<!-- claim: answer -->42<!-- /claim -->\n"
        "<!-- claim: answer -->41<!-- /claim -->\n"
        "<!-- claim: nope -->1<!-- /claim -->\n"
        "<!-- claim: answer -->forty<!-- /claim -->\n"
        "<!-- claim: answer lt -->50<!-- /claim -->\n"
        "<!-- claim: answer ge -->43<!-- /claim -->\n"
    )
    make_project(tmp_path, counting_claims(), {"value.txt": "42", "README.md": readme})
    assert run(tmp_path, "lock") == 0
    status, report = report_of(tmp_path, "check")
    assert status == 1
    assert results(report) == [
        (1, "pass"),
        (2, "fail"),
        (3, "unknown-claim"),
        (4, "bad-literal"),
        (5, "pass"),
        (6, "fail"),
    ]


def test_a_seeded_claims_tolerance_applies_to_printed_numbers(tmp_path: Path) -> None:
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('pi', kind='seeded', atol=0.01)\ndef pi():\n    return 3.14159\n"
    )
    readme = (
        "<!-- claim: pi -->3.14<!-- /claim -->\n"
        "<!-- claim: pi -->3.13<!-- /claim -->\n"
        "<!-- claim: pi -->3.12<!-- /claim -->\n"
        "<!-- claim: pi -->3<!-- /claim -->\n"
    )
    make_project(tmp_path, claims, {"README.md": readme})
    assert run(tmp_path, "lock") == 0
    status, report = report_of(tmp_path, "check")
    assert status == 1
    assert results(report) == [(1, "pass"), (2, "pass"), (3, "fail"), (4, "fail")]


def test_every_pattern_match_is_checked(tmp_path: Path) -> None:
    config = (
        DEFAULT_CONFIG + "[[bind]]\nfiles = ['README.md']\n"
        "pattern = '(\\d[\\d,]*) tests\\b'\nclaim = 'answer'\n"
    )
    readme = "We have 42 tests.\n\nElsewhere: 41 tests, and 42 tests.\n"
    make_project(tmp_path, counting_claims(), {"value.txt": "42", "README.md": readme}, config)
    assert run(tmp_path, "lock") == 0
    status, report = report_of(tmp_path, "check")
    assert status == 1
    assert results(report) == [(1, "pass"), (3, "fail"), (3, "pass")]


def test_exit_codes(tmp_path: Path) -> None:
    make_project(tmp_path, counting_claims(), {"value.txt": "42", "README.md": ANSWER_42})
    assert run(tmp_path, "list") == 0
    assert run(tmp_path, "lock") == 0
    assert run(tmp_path, "check") == 0
    write(tmp_path, "README.md", "<!-- claim: answer -->41<!-- /claim -->\n")
    assert run(tmp_path, "check") == 1
    write(tmp_path, "numeric-claim.toml", 'claims = ["claims.py"]\nbogus = true\n')
    assert run(tmp_path, "check") == 2


def test_list_shows_claims_and_occurrences_without_computing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = counting_claims(options=', paths=["src"], packages=["pkg"]')
    readme = "<!-- claim: answer -->42<!-- /claim -->\n\n<!-- claim: other lt -->7<!-- /claim -->\n"
    make_project(tmp_path, claims, {"README.md": readme, "value.txt": "42"})
    assert run(tmp_path, "list") == 0
    assert calls(tmp_path) == 0
    assert capsys.readouterr().out.splitlines() == [
        "claim answer [exact] packages=pkg paths=src reference=- nudged=- ladders=- jitter=-",
        "README.md:1 answer eq 42 quantity=value",
        "README.md:3 other lt 7 quantity=value",
        "1 claims, 2 occurrences",
    ]


def test_list_json_report(tmp_path: Path) -> None:
    make_project(tmp_path, counting_claims(), {"README.md": ANSWER_42, "value.txt": "42"})
    status, report = report_of(tmp_path, "list")
    assert status == 0
    assert report["command"] == "list"
    assert report["claims"] == [
        {
            "name": "answer",
            "kind": "exact",
            "packages": [],
            "paths": [],
            "reference": None,
            "nudged": [],
            "ladders": {},
            "jitter": {},
        },
    ]
    assert report["occurrences"] == [
        {
            "file": "README.md",
            "line": 1,
            "claim": "answer",
            "relation": "eq",
            "literal": "42",
            "origin": "sentinel",
            "quantity": "value",
        }
    ]


def test_the_lock_option_overrides_the_configured_path(tmp_path: Path) -> None:
    make_project(tmp_path, counting_claims(), {"value.txt": "42", "README.md": ANSWER_42})
    elsewhere = tmp_path / "locks" / "mine.json"
    elsewhere.parent.mkdir()
    assert run(tmp_path, "lock", "--lock", str(elsewhere)) == 0
    assert elsewhere.exists()
    assert not (tmp_path / "claims.lock.json").exists()
    assert run(tmp_path, "check", "--lock", str(elsewhere)) == 0
    assert run(tmp_path, "check") == 1


def test_the_configured_lock_path_is_relative_to_the_configuration_directory(
    tmp_path: Path,
) -> None:
    config = 'claims = ["claims.py"]\ndistributions = []\nlock = "locks/claims.json"\n'
    make_project(tmp_path, counting_claims(), {"value.txt": "42"}, config)
    (tmp_path / "locks").mkdir()
    assert run(tmp_path, "lock") == 0
    assert (tmp_path / "locks" / "claims.json").exists()


def test_the_root_option_separates_the_project_from_the_configuration(tmp_path: Path) -> None:
    configuration = tmp_path / "cfg"
    project = tmp_path / "proj"
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('size', paths=['data'])\ndef size(ctx):\n"
        "    return len((ctx.root / 'data' / 'a.txt').read_text())\n"
    )
    make_project(configuration, claims)
    write(project, "data/a.txt", "12345")
    write(project, "README.md", "<!-- claim: size -->5<!-- /claim -->\n")
    arguments = ["--config", str(configuration / "numeric-claim.toml"), "--root", str(project)]
    assert main(["lock", *arguments]) == 0
    assert main(["check", *arguments]) == 0
    write(project, "data/a.txt", "123456")
    assert main(["check", *arguments]) == 1
