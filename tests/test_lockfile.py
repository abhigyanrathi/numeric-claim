import json
from pathlib import Path

import pytest

from helpers import counting_claims, make_project, run, write
from numeric_claim import lockfile
from numeric_claim.errors import ConfigError
from numeric_claim.lockfile import LockEntry

FINGERPRINT = "f" * 64


def entry(value: int | float, kind: str = "seeded") -> LockEntry:
    if kind == "exact":
        return LockEntry(kind, value, FINGERPRINT, {"python_version": "3.12.0"})
    return LockEntry(kind, value, FINGERPRINT, {"python_version": "3.12.0"}, 1e-12, 0.0)


@pytest.mark.parametrize(
    "value",
    [0.1, 1 / 3, 5e-324, 1.7976931348623157e308, -0.0, 2.5, 1e-300, 123456789.123456789, -7e-9],
)
def test_floats_round_trip_exactly_through_hex(tmp_path: Path, value: float) -> None:
    path = tmp_path / "claims.lock.json"
    lockfile.write(path, {"c": entry(value)})
    restored = lockfile.read(path)["c"].value
    assert isinstance(restored, float)
    assert restored.hex() == value.hex()


def test_a_float_is_stored_as_hex_next_to_its_repr(tmp_path: Path) -> None:
    path = tmp_path / "claims.lock.json"
    lockfile.write(path, {"c": entry(0.1)})
    stored = json.loads(path.read_text(encoding="utf-8"))["claims"]["c"]
    assert stored["value"] == {"hex": (0.1).hex(), "repr": "0.1"}
    assert stored["rtol"] == 1e-12
    assert stored["atol"] == 0.0
    assert stored["kind"] == "seeded"


def test_an_exact_integer_is_stored_as_a_plain_number(tmp_path: Path) -> None:
    path = tmp_path / "claims.lock.json"
    lockfile.write(path, {"c": entry(264, "exact")})
    stored = json.loads(path.read_text(encoding="utf-8"))["claims"]["c"]
    assert stored["value"] == 264
    assert "rtol" not in stored
    assert lockfile.read(path)["c"].value == 264


def test_a_large_integer_survives(tmp_path: Path) -> None:
    path = tmp_path / "claims.lock.json"
    lockfile.write(path, {"c": entry(2**200 + 1, "exact")})
    assert lockfile.read(path)["c"].value == 2**200 + 1


def test_the_text_is_sorted_indented_and_ends_with_one_newline() -> None:
    entries = {"b": entry(2, "exact"), "a": entry(1, "exact")}
    expected = (
        "{\n"
        '  "claims": {\n'
        '    "a": {\n'
        '      "environment": {\n'
        '        "python_version": "3.12.0"\n'
        "      },\n"
        f'      "fingerprint": "{FINGERPRINT}",\n'
        '      "kind": "exact",\n'
        '      "value": 1\n'
        "    },\n"
        '    "b": {\n'
        '      "environment": {\n'
        '        "python_version": "3.12.0"\n'
        "      },\n"
        f'      "fingerprint": "{FINGERPRINT}",\n'
        '      "kind": "exact",\n'
        '      "value": 2\n'
        "    }\n"
        "  },\n"
        '  "format": 1\n'
        "}\n"
    )
    assert lockfile.dumps(entries) == expected


def test_the_file_uses_lf_line_endings_and_utf8(tmp_path: Path) -> None:
    path = tmp_path / "claims.lock.json"
    lockfile.write(path, {"c": entry(1, "exact")})
    data = path.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"}\n")
    data.decode("utf-8")


def test_a_missing_file_is_an_empty_lock(tmp_path: Path) -> None:
    assert lockfile.read(tmp_path / "absent.json") == {}


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"format": 2, "claims": {}}',
        '{"format": 1}',
        '{"format": 1, "claims": {"c": 3}}',
        '{"format": 1, "claims": {"c": {"kind": "exact", "fingerprint": "x", "environment": {}}}}',
        '{"format": 1, "claims": {"c": {"kind": "exact", "fingerprint": "x", "environment": {},'
        ' "value": true}}}',
        '{"format": 1, "claims": {"c": {"kind": "seeded", "fingerprint": "x", "environment": {},'
        ' "value": {"hex": "0x1p0", "repr": "1.0"}}}}',
        '{"format": 1, "claims": {"c": {"kind": "other", "fingerprint": "x", "environment": {},'
        ' "value": 1}}}',
        '{"format": 1, "claims": {"c": {"kind": "exact", "fingerprint": "x", "environment": {},'
        ' "value": {"hex": "garbage"}}}}',
    ],
)
def test_an_invalid_lock_file_is_a_configuration_error(tmp_path: Path, text: str) -> None:
    path = write(tmp_path, "claims.lock.json", text)
    with pytest.raises(ConfigError, match="lock file"):
        lockfile.read(path)


def test_two_runs_give_byte_identical_files(tmp_path: Path) -> None:
    make_project(tmp_path, counting_claims(kind="seeded", cast="float"), {"value.txt": "0.1"})
    assert run(tmp_path, "lock") == 0
    first = (tmp_path / "claims.lock.json").read_bytes()
    assert run(tmp_path, "lock", "--lock", str(tmp_path / "again.json")) == 0
    assert (tmp_path / "again.json").read_bytes() == first
    assert run(tmp_path, "lock") == 0
    assert (tmp_path / "claims.lock.json").read_bytes() == first


def test_claims_that_no_longer_exist_disappear_from_the_file(tmp_path: Path) -> None:
    two = (
        "from numeric_claim import claim\n\n\n"
        "@claim('keep')\ndef keep():\n    return 1\n\n\n"
        "@claim('drop')\ndef drop():\n    return 2\n"
    )
    make_project(tmp_path, two)
    assert run(tmp_path, "lock") == 0
    lock = tmp_path / "claims.lock.json"
    assert sorted(json.loads(lock.read_text(encoding="utf-8"))["claims"]) == ["drop", "keep"]
    write(tmp_path, "claims.py", two.split("\n\n\n@claim('drop')")[0] + "\n")
    assert run(tmp_path, "lock") == 0
    assert sorted(json.loads(lock.read_text(encoding="utf-8"))["claims"]) == ["keep"]
