import json
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy
import pytest

from helpers import make_project, run
from numeric_claim import ClaimContext, claim
from numeric_claim.claim import spec_of
from numeric_claim.engine import compute
from numeric_claim.errors import ClaimError
from numeric_claim.loader import LoadedClaim

CONTEXT = ClaimContext(Path("root"), Path("config"))
KINDS = ("exact", "seeded")


def loaded(function: Callable[..., Any], kind: str) -> LoadedClaim:
    decorated = claim("c", kind=kind)(function)
    spec = spec_of(decorated)
    assert spec is not None
    return LoadedClaim(spec, decorated, Path("claims.py"))


def numpy_claim(kind: str, expression: str) -> str:
    return (
        "import numpy\n\nfrom numeric_claim import claim\n\n\n"
        f"@claim('c', kind='{kind}')\ndef c():\n    return {expression}\n"
    )


def locked_value(root: Path) -> Any:
    lock = json.loads((root / "claims.lock.json").read_text(encoding="utf-8"))
    return lock["claims"]["c"]["value"]


@pytest.mark.parametrize("dtype", ["int64", "int32"])
def test_a_numpy_integer_from_an_exact_claim_is_locked_as_a_plain_json_integer(
    tmp_path: Path, dtype: str
) -> None:
    readme = "<!-- claim: c -->264<!-- /claim -->\n"
    make_project(tmp_path, numpy_claim("exact", f"numpy.{dtype}(264)"), {"README.md": readme})
    assert run(tmp_path, "lock") == 0
    value = locked_value(tmp_path)
    assert type(value) is int
    assert value == 264
    assert '"value": 264\n' in (tmp_path / "claims.lock.json").read_text(encoding="utf-8")
    report = tmp_path / "report.json"
    assert run(tmp_path, "check", "--recompute", "--json", str(report)) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["claims"][0]["value"] == 264


@pytest.mark.parametrize(
    ("dtype", "expected"), [("float32", 0.10000000149011612), ("float64", 0.1)]
)
def test_a_numpy_float_from_a_seeded_claim_is_stored_as_the_float_of_its_value(
    tmp_path: Path, dtype: str, expected: float
) -> None:
    make_project(tmp_path, numpy_claim("seeded", f"numpy.{dtype}(0.1)"))
    assert run(tmp_path, "lock") == 0
    assert locked_value(tmp_path) == {"hex": expected.hex(), "repr": repr(expected)}
    assert run(tmp_path, "check", "--recompute") == 0


@pytest.mark.parametrize("scalar", [numpy.int64(7), numpy.int32(7), numpy.uint8(7)])
def test_a_numpy_integer_is_an_int_when_exact_and_a_float_when_seeded(scalar: object) -> None:
    exact = compute(loaded(lambda: scalar, "exact"), CONTEXT)
    seeded = compute(loaded(lambda: scalar, "seeded"), CONTEXT)
    assert type(exact) is int
    assert exact == 7
    assert type(seeded) is float
    assert seeded == 7.0


@pytest.mark.parametrize("kind", KINDS)
def test_a_numpy_float_is_a_float_for_both_kinds(kind: str) -> None:
    value = compute(loaded(lambda: numpy.float32(0.1), kind), CONTEXT)
    assert type(value) is float
    assert value == 0.10000000149011612


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", [True, False, numpy.bool_(True), numpy.bool_(False)])
def test_booleans_are_claim_errors_for_both_kinds(kind: str, value: object) -> None:
    with pytest.raises(ClaimError, match="claim returned bool"):
        compute(loaded(lambda: value, kind), CONTEXT)


def test_a_fraction_from_a_seeded_claim_becomes_a_float() -> None:
    value = compute(loaded(lambda: Fraction(1, 3), "seeded"), CONTEXT)
    assert type(value) is float
    assert value == 1 / 3


@pytest.mark.parametrize("kind", KINDS)
def test_a_decimal_is_still_a_claim_error(kind: str) -> None:
    with pytest.raises(ClaimError, match="claim returned Decimal"):
        compute(loaded(lambda: Decimal("0.1"), kind), CONTEXT)


@pytest.mark.parametrize(
    "value", [numpy.float64(numpy.nan), numpy.float32(numpy.inf), Fraction(10**400)]
)
def test_numbers_without_a_finite_float_are_claim_errors(value: object) -> None:
    with pytest.raises(ClaimError):
        compute(loaded(lambda: value, "seeded"), CONTEXT)
