import json
from pathlib import Path
from typing import Any

import pytest

from helpers import calls, json_report, lock_data, logging_claims, make_project, run, write
from numeric_claim import lockfile
from numeric_claim.budget import Budget, Rung
from numeric_claim.errors import ConfigError
from numeric_claim.lockfile import LockEntry

FINGERPRINT = "f" * 64
PROBES = 1 + 32 + 32 + 4
TOTAL = """
@claim(
    "total",
    kind="seeded",
    ladders={"width": [11, 12.5], "n": [20, 40]},
    jitter={"width": 1e-9},
)
def total(*, x=2.0, n=10, width=10.0):
    log(x=x, n=n, width=width)
    return x + 1.0 / n + width / 7.0
"""
PLAIN = """
@claim("plain", kind="seeded")
def plain():
    return 0.5
"""
COUNT = """
@claim("count")
def count():
    return 3
"""


def stored(value: float) -> dict[str, str]:
    return {"hex": value.hex(), "repr": repr(value)}


def rewrite_lock(root: Path, change: Any) -> None:
    path = root / "claims.lock.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document["claims"])
    write(root, "claims.lock.json", json.dumps(document, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def locked(tmp_path: Path) -> Path:
    make_project(tmp_path, logging_claims(TOTAL))
    assert run(tmp_path, "lock") == 0
    assert calls(tmp_path) == PROBES
    return tmp_path


def test_a_budget_is_stored_beside_the_value(locked: Path) -> None:
    entry = lock_data(locked)["total"]
    assert sorted(entry) == [
        "atol",
        "budget",
        "environment",
        "fingerprint",
        "kind",
        "rtol",
        "value",
    ]
    budget = entry["budget"]
    assert sorted(budget) == ["jitter", "ladders", "nudge"]
    for key in ("nudge", "jitter"):
        deviation = float.fromhex(budget[key]["hex"])
        assert deviation > 0
        assert budget[key] == stored(deviation)
    assert budget["ladders"] == {
        "n": [
            {"setting": 20, "value": stored(2.0 + 1.0 / 20 + 10.0 / 7.0)},
            {"setting": 40, "value": stored(2.0 + 1.0 / 40 + 10.0 / 7.0)},
        ],
        "width": [
            {"setting": 11, "value": stored(2.0 + 1.0 / 10 + 11.0 / 7.0)},
            {"setting": 12.5, "value": stored(2.0 + 1.0 / 10 + 12.5 / 7.0)},
        ],
    }
    assert [type(rung["setting"]) for rung in budget["ladders"]["n"]] == [int, int]
    assert [type(rung["setting"]) for rung in budget["ladders"]["width"]] == [int, float]


def test_a_budget_uses_the_sorted_keys_and_indentation_of_the_file(locked: Path) -> None:
    text = (locked / "claims.lock.json").read_text(encoding="utf-8")
    assert '      "budget": {\n        "jitter": {\n          "hex": "' in text
    assert (
        '        "ladders": {\n'
        '          "n": [\n'
        "            {\n"
        '              "setting": 20,\n'
        '              "value": {\n'
        '                "hex": "'
    ) in text
    assert '              "setting": 12.5,\n' in text
    assert '        "nudge": {\n          "hex": "' in text
    assert text.endswith('    }\n  },\n  "format": 1\n}\n')


@pytest.mark.parametrize(
    ("options", "absent", "present"),
    [("", "jitter", "nudge"), (', jitter={"x": 1e-9}', "nudge", "jitter")],
)
def test_a_set_that_was_not_run_is_null_and_no_ladders_is_an_empty_object(
    tmp_path: Path, options: str, absent: str, present: str
) -> None:
    claims = f"@claim('c', kind='seeded'{options})\ndef c(*, x=1.5):\n    return x * x\n"
    make_project(tmp_path, logging_claims(claims))
    assert run(tmp_path, "lock") == 0
    budget = lock_data(tmp_path)["c"]["budget"]
    assert budget[absent] is None
    assert sorted(budget[present]) == ["hex", "repr"]
    assert budget["ladders"] == {}
    text = (tmp_path / "claims.lock.json").read_text(encoding="utf-8")
    assert f'        "{absent}": null' in text
    assert '        "ladders": {},\n' in text


@pytest.mark.parametrize(
    "budget",
    [
        Budget(
            5e-324,
            1 / 3,
            {
                "n": (Rung(20, 0.1), Rung(40, -2.5)),
                "width": (Rung(11, 1e300), Rung(12.5, 1.7976931348623157e308)),
            },
        ),
        Budget(None, 0.0, {}),
        Budget(0.1, None, {"n": (Rung(2**70, 1.0),)}),
    ],
)
def test_a_budget_is_read_back_exactly(tmp_path: Path, budget: Budget) -> None:
    entry = LockEntry("seeded", 0.1, FINGERPRINT, {"python_version": "3.12.0"}, 1e-12, 0.0, budget)
    path = tmp_path / "claims.lock.json"
    lockfile.write(path, {"c": entry})
    restored = lockfile.read(path)["c"]
    assert restored == entry
    assert restored.budget is not None
    for name, rungs in budget.ladders.items():
        kept = restored.budget.ladders[name]
        assert [type(rung.setting) for rung in kept] == [type(rung.setting) for rung in rungs]
        assert [rung.value.hex() for rung in kept] == [rung.value.hex() for rung in rungs]


def test_the_locked_budget_is_the_one_a_recomputation_reports(locked: Path) -> None:
    budget = lockfile.read(locked / "claims.lock.json")["total"].budget
    assert budget is not None
    status, report = json_report(locked, "check", "--recompute")
    assert status == 0
    reported = report["claims"][0]["budget"]
    assert (reported["nudge"], reported["jitter"]) == (budget.nudge, budget.jitter)
    assert reported["sigma"] == budget.sigma
    assert {
        name: [(rung["setting"], rung["value"]) for rung in rungs]
        for name, rungs in reported["ladders"].items()
    } == {
        name: [(rung.setting, rung.value) for rung in rungs]
        for name, rungs in budget.ladders.items()
    }


def test_two_runs_give_byte_identical_files(locked: Path) -> None:
    first = (locked / "claims.lock.json").read_bytes()
    assert run(locked, "lock", "--lock", str(locked / "again.json")) == 0
    assert (locked / "again.json").read_bytes() == first
    assert run(locked, "lock") == 0
    assert (locked / "claims.lock.json").read_bytes() == first
    assert b"\r" not in first


def test_the_fast_path_takes_the_value_and_the_budget_from_the_lock_and_calls_nothing(
    locked: Path,
) -> None:
    def tamper(claims: dict[str, Any]) -> None:
        claims["total"]["budget"]["nudge"] = stored(0.125)
        claims["total"]["budget"]["ladders"]["n"][1]["value"] = stored(7.5)

    rewrite_lock(locked, tamper)
    status, report = json_report(locked, "check")
    assert status == 0
    assert calls(locked) == PROBES
    claim = report["claims"][0]
    assert (claim["status"], claim["source"], claim["note"]) == ("ok", "lock", "")
    assert claim["budget"]["nudge"] == 0.125
    assert claim["budget"]["sigma"] == 0.125
    assert claim["budget"]["ladders"]["n"][1] == {"setting": 40, "value": 7.5, "error": None}


def test_a_budgeted_claim_whose_entry_lacks_a_budget_is_computed(locked: Path) -> None:
    before = lock_data(locked)["total"]["budget"]
    rewrite_lock(locked, lambda claims: claims["total"].pop("budget"))
    status, report = json_report(locked, "check")
    assert status == 0
    assert calls(locked) == 2 * PROBES
    claim = report["claims"][0]
    assert (claim["status"], claim["source"]) == ("ok", "computed")
    assert claim["note"] == (
        "the lock entry has no error budget; run `numeric-claim lock` to update the lock"
    )
    assert claim["budget"]["nudge"] == float.fromhex(before["nudge"]["hex"])
    assert "budget" not in lock_data(locked)["total"]


def drop_ladder(claims: dict[str, Any]) -> None:
    del claims["total"]["budget"]["ladders"]["n"]


def other_setting(claims: dict[str, Any]) -> None:
    claims["total"]["budget"]["ladders"]["n"][0]["setting"] = 30


def no_jitter(claims: dict[str, Any]) -> None:
    claims["total"]["budget"]["jitter"] = None


def extra_ladder(claims: dict[str, Any]) -> None:
    claims["total"]["budget"]["ladders"]["x"] = [{"setting": 3.0, "value": stored(1.0)}]


@pytest.mark.parametrize("change", [drop_ladder, other_setting, no_jitter, extra_ladder])
def test_a_locked_budget_that_is_not_what_the_probes_measure_is_computed(
    locked: Path, change: Any
) -> None:
    rewrite_lock(locked, change)
    status, report = json_report(locked, "check")
    assert status == 0
    assert calls(locked) == 2 * PROBES
    assert report["claims"][0]["source"] == "computed"
    assert sorted(report["claims"][0]["budget"]["ladders"]) == ["n", "width"]


def test_drift_compares_values_only(locked: Path) -> None:
    def tamper(claims: dict[str, Any]) -> None:
        claims["total"]["budget"]["nudge"] = stored(0.125)
        claims["total"]["budget"]["jitter"] = stored(0.25)
        claims["total"]["budget"]["ladders"]["n"][0]["value"] = stored(7.5)

    rewrite_lock(locked, tamper)
    status, report = json_report(locked, "check", "--recompute")
    assert status == 0
    assert report["claims"][0]["status"] == "ok"
    assert report["claims"][0]["budget"]["nudge"] != 0.125

    def move(claims: dict[str, Any]) -> None:
        claims["total"]["value"] = stored(3.0)

    rewrite_lock(locked, move)
    status, report = json_report(locked, "check", "--recompute")
    assert status == 1
    assert report["claims"][0]["status"] == "drift"


def test_entries_of_claims_without_a_budget_are_unchanged(tmp_path: Path) -> None:
    make_project(tmp_path, logging_claims(TOTAL, PLAIN, COUNT))
    assert run(tmp_path, "lock") == 0
    data = lock_data(tmp_path)
    assert sorted(data["plain"]) == ["atol", "environment", "fingerprint", "kind", "rtol", "value"]
    assert sorted(data["count"]) == ["environment", "fingerprint", "kind", "value"]
    assert "budget" in data["total"]
    document = json.loads((tmp_path / "claims.lock.json").read_text(encoding="utf-8"))
    assert document["format"] == 1
    entries = lockfile.read(tmp_path / "claims.lock.json")
    assert entries["plain"].budget is None
    assert entries["count"].budget is None
    plain = {name: entry for name, entry in entries.items() if name != "total"}
    assert "budget" not in lockfile.dumps(plain)


DEVIATION = '{"hex": "0x1p-3", "repr": "0.125"}'
VALUE = '{"hex": "0x1p0", "repr": "1.0"}'


@pytest.mark.parametrize(
    "budget",
    [
        "3",
        "[]",
        '{"nudge": null, "jitter": null, "ladders": {}}',
        f'{{"nudge": {DEVIATION}, "jitter": null}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": []}}',
        '{"nudge": "0.125", "jitter": null, "ladders": {}}',
        f'{{"nudge": null, "jitter": {DEVIATION}, "ladders": {{"n": {{}}}}}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": {{"n": [3]}}}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": {{"n": [{{"value": {VALUE}}}]}}}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": '
        f'{{"n": [{{"setting": true, "value": {VALUE}}}]}}}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": '
        f'{{"n": [{{"setting": "20", "value": {VALUE}}}]}}}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": '
        '{"n": [{"setting": 20, "value": "1.0"}]}}',
        f'{{"nudge": {DEVIATION}, "jitter": null, "ladders": {{"n": [{{"setting": 20}}]}}}}',
    ],
)
def test_an_invalid_budget_is_a_configuration_error(tmp_path: Path, budget: str) -> None:
    text = (
        '{"format": 1, "claims": {"c": {"kind": "seeded", "fingerprint": "x", "environment": {}, '
        f'"value": {VALUE}, "rtol": 1e-12, "atol": 0.0, "budget": {budget}}}}}}}'
    )
    assert json.loads(text)["claims"]["c"]["budget"] is not None
    path = write(tmp_path, "claims.lock.json", text)
    with pytest.raises(ConfigError, match="lock file"):
        lockfile.read(path)


def test_a_null_budget_reads_as_no_budget(tmp_path: Path) -> None:
    text = (
        '{"format": 1, "claims": {"c": {"kind": "seeded", "fingerprint": "x", "environment": {}, '
        f'"value": {VALUE}, "rtol": 1e-12, "atol": 0.0, "budget": null}}}}}}'
    )
    path = write(tmp_path, "claims.lock.json", text)
    assert lockfile.read(path)["c"].budget is None
