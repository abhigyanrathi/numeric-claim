from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from numeric_claim import ClaimContext, claim
from numeric_claim.claim import spec_of
from numeric_claim.engine import compute
from numeric_claim.errors import ClaimError
from numeric_claim.loader import LoadedClaim

CONTEXT = ClaimContext(Path("root"), Path("config"))


def loaded(function: Callable[..., Any], kind: str = "exact") -> LoadedClaim:
    decorated = claim("c", kind=kind)(function)
    spec = spec_of(decorated)
    assert spec is not None
    return LoadedClaim(spec, decorated, Path("claims.py"))


def test_decorator_returns_the_function_with_metadata() -> None:
    def original() -> int:
        return 1

    decorated = claim("n", paths=["src"], packages=("pkg",))(original)
    assert decorated is original
    spec = spec_of(original)
    assert spec is not None
    assert (spec.name, spec.kind, spec.paths, spec.packages) == ("n", "exact", ("src",), ("pkg",))
    assert original() == 1


def test_exact_claim_has_no_tolerance() -> None:
    spec = spec_of(claim("n")(lambda: 1))
    assert spec is not None
    assert spec.tolerance(1000) == 0.0


def test_seeded_defaults_and_tolerance() -> None:
    spec = spec_of(claim("s", kind="seeded")(lambda: 1.0))
    assert spec is not None
    assert (spec.rtol, spec.atol) == (1e-12, 0.0)
    assert spec.tolerance(-2.0) == 2e-12


def test_seeded_tolerance_uses_the_larger_of_atol_and_rtol() -> None:
    spec = spec_of(claim("s", kind="seeded", rtol=1e-3, atol=0.5)(lambda: 1.0))
    assert spec is not None
    assert spec.tolerance(10.0) == 0.5
    assert spec.tolerance(1000.0) == 1.0


@pytest.mark.parametrize("options", [{"rtol": 1e-6}, {"atol": 1e-6}, {"rtol": 0.0, "atol": 0.0}])
def test_exact_claim_rejects_tolerances(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="exact"):
        claim("n", kind="exact", **options)


@pytest.mark.parametrize("kind", ["approx", "", "EXACT"])
def test_unknown_kind_raises_when_the_decorator_runs(kind: str) -> None:
    with pytest.raises(ValueError, match="kind"):
        claim("n", kind=kind)


@pytest.mark.parametrize(
    "name", ["", "1abc", "has space", "a/b", "caf\N{LATIN SMALL LETTER E WITH ACUTE}", "trailing\n"]
)
def test_invalid_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        claim(name)


@pytest.mark.parametrize("name", ["a", "_x", "test_count", "a.b-c_d", "X9"])
def test_valid_names_are_accepted(name: str) -> None:
    claim(name)


@pytest.mark.parametrize(
    "options", [{"rtol": -1.0}, {"atol": float("nan")}, {"rtol": float("inf")}]
)
def test_seeded_rejects_unusable_tolerances(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="finite"):
        claim("s", kind="seeded", **options)


def test_a_single_string_is_not_a_list_of_paths() -> None:
    with pytest.raises(ValueError, match="paths"):
        claim("n", paths="src")


def test_compute_calls_a_function_without_arguments() -> None:
    assert compute(loaded(lambda: 7), CONTEXT) == 7


def test_compute_passes_the_context() -> None:
    seen: list[ClaimContext] = []

    def takes_context(ctx: ClaimContext) -> int:
        seen.append(ctx)
        return 3

    assert compute(loaded(takes_context), CONTEXT) == 3
    assert seen == [CONTEXT]


def test_exact_float_stays_a_float_and_seeded_int_becomes_one() -> None:
    exact = compute(loaded(lambda: 2.5), CONTEXT)
    seeded = compute(loaded(lambda: 3, kind="seeded"), CONTEXT)
    assert exact == 2.5
    assert isinstance(seeded, float)
    assert seeded == 3.0


def raises_runtime_error() -> int:
    raise RuntimeError("boom")


def raises_system_exit() -> int:
    raise SystemExit(0)


def two_arguments(a: object, b: object) -> int:
    return 1


@pytest.mark.parametrize(
    "function",
    [
        lambda: True,
        lambda: "264",
        lambda: None,
        lambda: float("nan"),
        lambda: float("inf"),
        lambda: -float("inf"),
        lambda: [1],
        raises_runtime_error,
        raises_system_exit,
        two_arguments,
    ],
)
def test_bad_returns_and_exceptions_are_claim_errors(function: Callable[..., Any]) -> None:
    with pytest.raises(ClaimError):
        compute(loaded(function), CONTEXT)


def test_a_claim_error_names_the_exception() -> None:
    with pytest.raises(ClaimError, match="RuntimeError: boom"):
        compute(loaded(raises_runtime_error), CONTEXT)
