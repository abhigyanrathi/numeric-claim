import hashlib
import math
import random
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from helpers import calls, json_report, logged, logging_claims, make_project, run
from numeric_claim import ClaimContext, claim
from numeric_claim.budget import Budget, Rung, draws
from numeric_claim.claim import spec_of
from numeric_claim.engine import measure
from numeric_claim.errors import ClaimError
from numeric_claim.loader import LoadedClaim

CONTEXT = ClaimContext(Path("root"), Path("config"))
DEFAULTS: dict[str, float] = {"x": 2.0, "y": -3.0, "zero": 0.0, "count": 5, "n": 10, "width": 10.0}
LADDERS = {"n": [20, 40], "width": [11, 12.5]}
JITTER = {"width": 1e-9}

AREA = """
@claim("area", kind="seeded"{options})
def area(*, x=2.0, y=-3.0, zero=0.0, count=5, n=10, width=10.0):
    log(x=x, y=y, zero=zero, count=count, n=n, width=width)
    return x * y + zero + count + 1.0 / n + width / 7.0
"""


def formula(x: float, y: float, zero: float, count: float, n: float, width: float) -> float:
    return x * y + zero + count + 1.0 / n + width / 7.0


def recording(name: str = "area", **options: Any) -> tuple[LoadedClaim, list[dict[str, float]]]:
    seen: list[dict[str, float]] = []

    def area(
        *,
        x: float = 2.0,
        y: float = -3.0,
        zero: float = 0.0,
        count: int = 5,
        n: int = 10,
        width: float = 10.0,
    ) -> float:
        seen.append({"x": x, "y": y, "zero": zero, "count": count, "n": n, "width": width})
        return formula(x, y, zero, count, n, width)

    decorated = claim(name, kind="seeded", **options)(area)
    spec = spec_of(decorated)
    assert spec is not None
    return LoadedClaim(spec, decorated, Path("claims.py")), seen


def measured(name: str = "area", **options: Any) -> tuple[Budget, list[dict[str, float]]]:
    loaded, seen = recording(name, **options)
    value, budget = measure(loaded, CONTEXT)
    assert value == formula(**DEFAULTS)
    assert budget is not None
    return budget, seen


def nearby(default: float) -> set[float]:
    return {
        math.nextafter(default, toward, steps=steps)
        for toward in (math.inf, -math.inf)
        for steps in range(1, 9)
    }


def specified_draws(
    name: str, nudged: dict[str, float], jittered: dict[str, tuple[float, float]]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    nudges: list[dict[str, float]] = []
    for _ in range(32 if nudged else 0):
        call = {}
        for parameter, default in nudged.items():
            k = 1 + int(8 * rng.random())
            toward = math.inf if rng.random() < 0.5 else -math.inf
            call[parameter] = math.nextafter(default, toward, steps=k)
        nudges.append(call)
    jitters: list[dict[str, float]] = []
    for _ in range(32 if jittered else 0):
        call = {}
        for parameter, (default, a) in jittered.items():
            u = 2.0 * rng.random() - 1.0
            call[parameter] = default * (1.0 + a * u)
        jitters.append(call)
    return nudges, jitters


def test_the_first_call_uses_the_defaults() -> None:
    _, seen = measured(ladders=LADDERS, jitter=JITTER)
    assert seen[0] == DEFAULTS


def test_each_nudged_input_moves_by_1_to_8_units_in_the_last_place_and_nothing_else_moves() -> None:
    _, seen = measured(ladders=LADDERS, jitter=JITTER)
    nudged = seen[1:33]
    for call in nudged:
        assert call["x"] in nearby(2.0)
        assert call["y"] in nearby(-3.0)
        assert {name: call[name] for name in ("zero", "count", "n", "width")} == {
            "zero": 0.0,
            "count": 5,
            "n": 10,
            "width": 10.0,
        }
        assert type(call["count"]) is int
    for name in ("x", "y"):
        moved = {call[name] for call in nudged}
        assert len(moved) > 8
        assert min(moved) < DEFAULTS[name] < max(moved)


def test_each_jittered_value_lies_within_its_relative_amplitude_and_nothing_else_moves() -> None:
    _, seen = measured(ladders=LADDERS, jitter=JITTER)
    jittered = seen[33:65]
    for call in jittered:
        assert abs(call["width"] - 10.0) <= 10.0 * 1e-9 * (1 + 1e-12)
        assert {name: call[name] for name in call if name != "width"} == {
            name: DEFAULTS[name] for name in DEFAULTS if name != "width"
        }
    widths = {call["width"] for call in jittered}
    assert len(widths) == 32
    assert min(widths) < 10.0 < max(widths)


def test_each_ladder_setting_is_called_once_with_only_that_parameter_changed() -> None:
    budget, seen = measured(ladders={"width": [11, 12.5], "n": [20, 40]}, jitter=JITTER)
    assert seen[65:] == [
        {**DEFAULTS, "n": 20},
        {**DEFAULTS, "n": 40},
        {**DEFAULTS, "width": 11.0},
        {**DEFAULTS, "width": 12.5},
    ]
    assert [type(call["n"]) for call in seen[65:67]] == [int, int]
    assert [type(call["width"]) for call in seen[67:]] == [float, float]
    assert budget.ladders == {
        "n": (
            Rung(20, formula(**{**DEFAULTS, "n": 20})),
            Rung(40, formula(**{**DEFAULTS, "n": 40})),
        ),
        "width": (
            Rung(11, formula(**{**DEFAULTS, "width": 11.0})),
            Rung(12.5, formula(**{**DEFAULTS, "width": 12.5})),
        ),
    }
    assert list(budget.ladders) == ["n", "width"]


def test_the_calls_are_those_the_rules_specify() -> None:
    _, seen = measured(ladders=LADDERS, jitter=JITTER)
    nudges, jitters = specified_draws("area", {"x": 2.0, "y": -3.0}, {"width": (10.0, 1e-9)})
    assert seen[1:33] == [{**DEFAULTS, **call} for call in nudges]
    assert seen[33:65] == [{**DEFAULTS, **call} for call in jitters]


def test_nudge_jitter_and_sigma_are_the_standard_deviations_of_the_specified_draws() -> None:
    budget, _ = measured(ladders=LADDERS, jitter=JITTER)
    nudges, jitters = specified_draws("area", {"x": 2.0, "y": -3.0}, {"width": (10.0, 1e-9)})
    nudge = statistics.stdev(formula(**{**DEFAULTS, **call}) for call in nudges)
    jitter = statistics.stdev(formula(**{**DEFAULTS, **call}) for call in jitters)
    assert budget.nudge == nudge
    assert budget.jitter == jitter
    assert 0 < nudge < jitter
    assert budget.sigma == jitter


def test_sigma_is_the_larger_deviation_whichever_it_is() -> None:
    assert Budget(0.5, 0.25, {}).sigma == 0.5
    assert Budget(0.25, 0.5, {}).sigma == 0.5
    assert Budget(None, 0.25, {}).sigma == 0.25
    assert Budget(0.25, None, {}).sigma == 0.25


def test_a_set_that_is_not_declared_is_not_run() -> None:
    only_nudged, seen = measured()
    assert len(seen) == 33
    assert only_nudged.jitter is None
    assert only_nudged.nudge is not None
    assert only_nudged.sigma == only_nudged.nudge
    assert only_nudged.ladders == {}
    names = ("x", "y", "width")
    only_jittered, seen = measured(jitter=dict.fromkeys(names, 1e-9))
    assert len(seen) == 33
    assert only_jittered.nudge is None
    assert only_jittered.sigma == only_jittered.jitter
    for call in seen[1:]:
        assert all(call[name] != DEFAULTS[name] for name in names)


def test_the_same_claim_name_gives_the_same_draws_and_another_name_gives_others() -> None:
    first, seen = measured(ladders=LADDERS, jitter=JITTER)
    second, again = measured(ladders=LADDERS, jitter=JITTER)
    assert again == seen
    assert second == first
    other, different = measured("volume", ladders=LADDERS, jitter=JITTER)
    assert different[0] == seen[0]
    assert different[1:33] != seen[1:33]
    assert different[33:65] != seen[33:65]
    assert different[65:] == seen[65:]
    assert (other.nudge, other.jitter) != (first.nudge, first.jitter)


def test_draws_are_empty_for_a_set_with_nothing_declared() -> None:
    spec = recording()[0].spec
    assert spec.probes is not None
    nudged, jittered = draws("area", spec.probes)
    assert len(nudged) == 32
    assert jittered == []
    assert all(set(call) == {"x", "y", "width"} for call in nudged)


def test_the_context_is_passed_positionally_to_every_call() -> None:
    seen: list[tuple[ClaimContext, int]] = []

    def function(ctx: ClaimContext, *, x: float = 1.5, n: int = 1) -> float:
        seen.append((ctx, n))
        return x

    decorated = claim("c", kind="seeded", ladders={"n": [2]})(function)
    spec = spec_of(decorated)
    assert spec is not None
    value, budget = measure(LoadedClaim(spec, decorated, Path("claims.py")), CONTEXT)
    assert value == 1.5
    assert budget is not None
    assert len(seen) == 34
    assert all(ctx is CONTEXT for ctx, _ in seen)
    assert seen[-1] == (CONTEXT, 2)


def test_a_claim_without_keyword_only_parameters_is_called_once_and_has_no_budget() -> None:
    seen: list[int] = []

    def function() -> float:
        seen.append(1)
        return 0.5

    decorated = claim("c", kind="seeded", reference="0.5")(function)
    spec = spec_of(decorated)
    assert spec is not None
    assert measure(LoadedClaim(spec, decorated, Path("claims.py")), CONTEXT) == (0.5, None)
    assert seen == [1]


def boom() -> float:
    raise RuntimeError("boom")


def leave() -> float:
    raise SystemExit(3)


@pytest.mark.parametrize(
    ("fails", "result", "message"),
    [
        (lambda x, n, w: x != 1.5, boom, r"probe with x=\S+: RuntimeError: boom"),
        (lambda x, n, w: w != 1.0, leave, r"probe with w=\S+: SystemExit: 3"),
        (lambda x, n, w: n == 40, boom, "probe with n=40: RuntimeError: boom"),
        (lambda x, n, w: n == 40, lambda: math.nan, "probe with n=40: claim returned a non-finite"),
        (lambda x, n, w: n == 20, lambda: "x", "probe with n=20: claim returned str"),
        (lambda x, n, w: n == 20, lambda: True, "probe with n=20: claim returned bool"),
    ],
)
def test_a_probe_call_that_raises_or_returns_an_invalid_value_is_a_claim_error(
    fails: Callable[[float, int, float], bool], result: Callable[[], Any], message: str
) -> None:
    def function(*, x: float = 1.5, n: int = 10, w: float = 1.0) -> Any:
        return result() if fails(x, n, w) else x

    decorated = claim("c", kind="seeded", ladders={"n": [20, 40]}, jitter={"w": 1e-9})(function)
    spec = spec_of(decorated)
    assert spec is not None
    with pytest.raises(ClaimError, match=message):
        measure(LoadedClaim(spec, decorated, Path("claims.py")), CONTEXT)


def test_a_claim_whose_probe_call_raises_is_in_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('c', kind='seeded', ladders={'n': [20]})\n"
        "def c(*, x=1.5, n=10):\n"
        "    if n == 20:\n"
        "        raise RuntimeError('boom')\n"
        "    return x\n"
    )
    make_project(tmp_path, claims, {"README.md": "<!-- claim: c -->1.5<!-- /claim -->\n"})
    assert run(tmp_path, "lock") == 1
    assert "error c: probe with n=20: RuntimeError: boom" in capsys.readouterr().out
    assert not (tmp_path / "claims.lock.json").exists()
    status, report = json_report(tmp_path, "check")
    assert status == 1
    assert report["claims"][0]["status"] == "error"
    assert report["claims"][0]["note"] == "probe with n=20: RuntimeError: boom"
    assert report["claims"][0]["budget"] is None
    assert [item["result"] for item in report["occurrences"]] == ["claim-error"]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ("", 33),
        (', jitter={"x": 1e-9, "y": 1e-9, "width": 1e-9}', 33),
        (', jitter={"width": 1e-9}', 65),
        (', ladders={"n": [20, 40, 80]}', 36),
        (', ladders={"n": [20, 40], "width": [11, 12.5]}, jitter={"width": 1e-9}', 69),
    ],
)
def test_the_number_of_calls_is_exact_and_the_fast_path_makes_none(
    tmp_path: Path, options: str, expected: int
) -> None:
    make_project(tmp_path, logging_claims(AREA.format(options=options)))
    assert run(tmp_path, "lock") == 0
    assert calls(tmp_path) == expected
    assert logged(tmp_path)[0] == DEFAULTS
    assert run(tmp_path, "check") == 0
    assert calls(tmp_path) == expected
    assert run(tmp_path, "check", "--recompute") == 0
    assert calls(tmp_path) == 2 * expected
    assert run(tmp_path, "list") == 0
    assert calls(tmp_path) == 2 * expected


def test_a_claim_without_a_budget_is_still_called_once(tmp_path: Path) -> None:
    plain = """
    @claim("plain", kind="seeded", reference="0.5")
    def plain():
        log()
        return 0.5
    """
    make_project(tmp_path, logging_claims(plain))
    assert run(tmp_path, "lock") == 0
    assert calls(tmp_path) == 1
    assert run(tmp_path, "check") == 0
    assert calls(tmp_path) == 1
    assert run(tmp_path, "check", "--recompute") == 0
    assert calls(tmp_path) == 2
