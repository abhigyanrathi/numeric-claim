import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scipy import stats

from helpers import (
    DEFAULT_CONFIG,
    calls,
    json_report,
    lock_data,
    logged,
    logging_claims,
    make_project,
    run,
    write,
)
from numeric_claim import ClaimContext, claim, lockfile
from numeric_claim.claim import spec_of
from numeric_claim.config import load_config
from numeric_claim.engine import measure, sample
from numeric_claim.errors import ClaimError, ConfigError
from numeric_claim.loader import LoadedClaim
from numeric_claim.lockfile import LockEntry
from numeric_claim.montecarlo import Summary, holm

SOURCE = Path(__file__).resolve().parents[1] / "src"
CONTEXT = ClaimContext(Path("root"), Path("config"))
UNIFORM = """
import random


@claim("u", kind="random", replicates={replicates}{options})
def u(ctx):
    log(seed=ctx.seed)
    return random.Random(ctx.seed).random()
"""
GAUSS = """
import random


@claim("g", kind="random"{options})
def g(ctx):
    return random.Random(ctx.seed).gauss(10.0, 1.0)
"""
CONSTANT = """
@claim("c", kind="random", replicates=8)
def c(ctx):
    return 2.5
"""
FAILING = """
SEEN = []


@claim("f", kind="random", replicates=8)
def f(ctx):
    SEEN.append(ctx.seed)
    if len(SEEN) == 3:
        raise RuntimeError("boom")
    return 1.0
"""
NOTE = (
    "the lock entry has a different number of replicates; "
    "run `numeric-claim lock` to update the lock"
)


def specified_seeds(name: str, count: int) -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"{name}/{index}".encode()).digest()[:8], "big")
        for index in range(count)
    ]


def uniform(replicates: int, options: str = "") -> str:
    return logging_claims(UNIFORM.format(replicates=replicates, options=options))


def uniform_values(replicates: int) -> list[float]:
    return [random.Random(seed).random() for seed in specified_seeds("u", replicates)]


def gauss(options: str = "") -> str:
    return logging_claims(GAUSS.format(options=options))


def gauss_summary(replicates: int = 32) -> Summary:
    values = [random.Random(seed).gauss(10.0, 1.0) for seed in specified_seeds("g", replicates)]
    return Summary(statistics.fmean(values), statistics.stdev(values), replicates)


def sentinel(literal: str, relation: str = "", name: str = "g") -> str:
    return f"<!-- claim: {name} {relation} -->{literal}<!-- /claim -->\n"


def stored(value: float) -> dict[str, str]:
    return {"hex": value.hex(), "repr": repr(value)}


def rewrite_lock(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    document = json.loads((root / "claims.lock.json").read_text(encoding="utf-8"))
    change(document["claims"])
    write(root, "claims.lock.json", json.dumps(document, indent=2, sort_keys=True) + "\n")


def loaded(function: Callable[..., Any], kind: str = "random", **options: Any) -> LoadedClaim:
    decorated = claim("r", kind=kind, **options)(function)
    spec = spec_of(decorated)
    assert spec is not None
    return LoadedClaim(spec, decorated, Path("claims.py"))


def one_context(ctx: ClaimContext) -> float:
    return 1.0


def test_a_random_claim_declares_its_replicates_and_tolerances() -> None:
    spec = loaded(one_context).spec
    assert (spec.kind, spec.replicates, spec.rtol, spec.atol) == ("random", 32, 0.0, 0.0)
    assert (spec.reference, spec.probes, spec.tolerance(-20.0)) == (None, None, 0.0)
    options: dict[str, Any] = {"rtol": 1e-3, "atol": 0.5, "packages": ["pkg"], "paths": ["src"]}
    spec = loaded(one_context, replicates=8, **options).spec
    assert (spec.replicates, spec.rtol, spec.atol) == (8, 1e-3, 0.5)
    assert (spec.packages, spec.paths, spec.tolerance(-20.0)) == (("pkg",), ("src",), 0.5)
    assert loaded(one_context, rtol=0.1).spec.tolerance(-20.0) == 2.0
    assert loaded(one_context, replicates=100_000).spec.replicates == 100_000


@pytest.mark.parametrize(
    "function",
    [lambda ctx: 1.0, lambda ctx, /: 1.0, lambda ctx=None: 1.0, lambda ctx, *rest, **extra: 1.0],
)
def test_a_random_claim_takes_one_positional_context(function: Callable[..., Any]) -> None:
    assert loaded(function).spec.kind == "random"


INVALID: list[tuple[dict[str, Any], Callable[..., Any], str]] = [
    ({"replicates": 7}, one_context, "replicates must be an int from 8 to 100000, got 7"),
    ({"replicates": 100_001}, one_context, "an int from 8 to 100000, got 100001"),
    ({"replicates": 32.0}, one_context, "replicates must be an int from 8 to 100000, got 32.0"),
    ({"replicates": True}, one_context, "replicates must be an int from 8 to 100000, got True"),
    ({"replicates": "32"}, one_context, "replicates must be an int from 8 to 100000, got '32'"),
    ({"reference": "1.0"}, one_context, "a random claim takes no reference, ladders or jitter"),
    ({"ladders": {}}, one_context, "a random claim takes no reference, ladders or jitter"),
    ({"jitter": {"x": 1e-9}}, one_context, "a random claim takes no reference, ladders or jitter"),
    ({"rtol": -1.0}, one_context, "rtol must be a finite number >= 0"),
    ({"atol": math.inf}, one_context, "atol must be a finite number >= 0"),
    ({}, lambda: 1.0, "a random claim takes one positional ClaimContext and no keyword-only"),
    ({}, lambda ctx, other: 1.0, "a random claim takes one positional ClaimContext"),
    ({}, lambda ctx, *, n=1: 1.0, "a random claim takes one positional ClaimContext"),
    ({}, lambda *, n=1: 1.0, "a random claim takes one positional ClaimContext"),
    ({}, lambda *args: 1.0, "a random claim takes one positional ClaimContext"),
]


@pytest.mark.parametrize(("options", "function", "message"), INVALID)
def test_an_invalid_random_claim_raises_when_declared(
    options: dict[str, Any], function: Callable[..., Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        claim("r", kind="random", **options)(function)


@pytest.mark.parametrize("kind", ["exact", "seeded"])
def test_only_a_random_claim_takes_replicates(kind: str) -> None:
    message = f"only a random claim takes replicates, and this one is '{kind}'"
    with pytest.raises(ValueError, match=re.escape(message)):
        claim("c", kind=kind, replicates=32)
    claim("c", kind=kind, replicates=None)


def recorder(name: str) -> tuple[LoadedClaim, list[ClaimContext]]:
    seen: list[ClaimContext] = []

    def function(ctx: ClaimContext) -> float:
        seen.append(ctx)
        return 0.5

    decorated = claim(name, kind="random")(function)
    spec = spec_of(decorated)
    assert spec is not None
    return LoadedClaim(spec, decorated, Path("claims.py")), seen


def test_each_replicate_receives_its_specified_seed_and_the_rest_of_the_context() -> None:
    first, seen = recorder("r")
    sample(first, CONTEXT, 10)
    assert [ctx.seed for ctx in seen] == specified_seeds("r", 10)
    assert len({ctx.seed for ctx in seen}) == 10
    assert all((ctx.root, ctx.config_dir) == (CONTEXT.root, CONTEXT.config_dir) for ctx in seen)
    assert CONTEXT.seed is None
    again, repeated = recorder("r")
    sample(again, CONTEXT, 10)
    assert repeated == seen
    other, different = recorder("s")
    sample(other, CONTEXT, 10)
    assert not {ctx.seed for ctx in different} & {ctx.seed for ctx in seen}


def test_other_kinds_see_no_seed() -> None:
    seen: list[int | None] = []

    def exact(ctx: ClaimContext) -> int:
        seen.append(ctx.seed)
        return 1

    def seeded(ctx: ClaimContext, *, x: float = 1.5) -> float:
        seen.append(ctx.seed)
        return x

    measure(loaded(exact, "exact"), CONTEXT)
    measure(loaded(seeded, "seeded"), CONTEXT)
    assert len(seen) == 1 + 33
    assert set(seen) == {None}


def test_sample_summarizes_the_values_of_the_replicates() -> None:
    order = specified_seeds("r", 8)

    def rank(ctx: ClaimContext) -> int:
        assert ctx.seed is not None
        return order.index(ctx.seed)

    summary = sample(loaded(rank, replicates=8), CONTEXT, 8)
    assert summary == Summary(3.5, math.sqrt(6.0), 8)
    assert type(summary.mean) is float
    assert summary.se == math.sqrt(6.0) / math.sqrt(8)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (lambda: math.nan, "replicate 4: claim returned a non-finite value: nan"),
        (lambda: "1.0", "replicate 4: claim returned str"),
        (lambda: True, "replicate 4: claim returned bool"),
        (lambda: 1 / 0, "replicate 4: ZeroDivisionError: division by zero"),
    ],
)
def test_a_replicate_that_raises_or_returns_an_invalid_value_is_a_claim_error(
    result: Callable[[], Any], message: str
) -> None:
    order = specified_seeds("r", 8)

    def function(ctx: ClaimContext) -> Any:
        assert ctx.seed is not None
        return result() if order.index(ctx.seed) == 4 else 1.0

    with pytest.raises(ClaimError, match=re.escape(message)):
        sample(loaded(function, replicates=8), CONTEXT, 8)


def test_a_replicate_that_raises_puts_the_claim_in_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_project(tmp_path, logging_claims(FAILING), {"README.md": sentinel("1.0", name="f")})
    assert run(tmp_path, "lock") == 1
    assert "error f: replicate 2: RuntimeError: boom" in capsys.readouterr().out
    assert not (tmp_path / "claims.lock.json").exists()
    status, report = json_report(tmp_path, "check")
    assert status == 1
    found = report["claims"][0]
    assert (found["status"], found["note"]) == ("error", "replicate 2: RuntimeError: boom")
    assert found["random"] is None
    (occurrence,) = report["occurrences"]
    assert occurrence["result"] == "claim-error"
    assert (occurrence["p"], occurrence["verdict"]) == (None, None)


@pytest.mark.parametrize("replicates", [8, 20])
def test_a_random_claim_is_called_once_per_replicate_and_not_at_all_on_the_fast_path(
    tmp_path: Path, replicates: int
) -> None:
    make_project(tmp_path, uniform(replicates))
    assert run(tmp_path, "lock") == 0
    assert calls(tmp_path) == replicates
    assert [call["seed"] for call in logged(tmp_path)] == specified_seeds("u", replicates)
    assert run(tmp_path, "check") == 0
    assert calls(tmp_path) == replicates
    assert run(tmp_path, "check", "--recompute") == 0
    assert calls(tmp_path) == 2 * replicates
    assert run(tmp_path, "list") == 0
    assert calls(tmp_path) == 2 * replicates


def test_the_fast_path_needs_the_declared_number_of_replicates(tmp_path: Path) -> None:
    make_project(tmp_path, uniform(8))
    assert run(tmp_path, "lock") == 0
    rewrite_lock(tmp_path, lambda claims: claims["u"].update(replicates=9))
    status, report = json_report(tmp_path, "check")
    assert status == 0
    assert calls(tmp_path) == 16
    found = report["claims"][0]
    assert (found["status"], found["source"], found["note"]) == ("ok", "computed", NOTE)
    write(tmp_path, "claims.py", uniform(12))
    status, report = json_report(tmp_path, "check")
    assert calls(tmp_path) == 28
    assert report["claims"][0]["source"] == "computed"
    assert report["claims"][0]["random"]["replicates"] == 12


def test_the_value_is_the_mean_of_the_replicates_and_their_summary_is_reported(
    tmp_path: Path,
) -> None:
    make_project(tmp_path, uniform(20))
    status, report = json_report(tmp_path, "check")
    assert status == 1
    values = uniform_values(20)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    found = report["claims"][0]
    assert (found["status"], found["value"]) == ("unlocked", mean)
    assert found["random"] == {"replicates": 20, "mean": mean, "sd": sd, "se": sd / math.sqrt(20)}


def test_the_lock_entry_holds_the_replicates_the_mean_and_the_sd(tmp_path: Path) -> None:
    make_project(tmp_path, uniform(8, ", rtol=1e-3, atol=0.25"))
    assert run(tmp_path, "lock") == 0
    entry = lock_data(tmp_path)["u"]
    assert sorted(entry) == [
        "atol",
        "environment",
        "fingerprint",
        "kind",
        "replicates",
        "rtol",
        "sd",
        "value",
    ]
    values = uniform_values(8)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    assert (entry["kind"], entry["replicates"], entry["rtol"], entry["atol"]) == (
        "random",
        8,
        1e-3,
        0.25,
    )
    assert (entry["value"], entry["sd"]) == (stored(mean), stored(sd))
    restored = lockfile.read(tmp_path / "claims.lock.json")["u"]
    summary = Summary(mean, sd, 8)
    fingerprint, environment = entry["fingerprint"], entry["environment"]
    assert restored == LockEntry(
        "random", mean, fingerprint, environment, 1e-3, 0.25, None, summary
    )


def test_two_lock_runs_give_byte_identical_files(tmp_path: Path) -> None:
    make_project(tmp_path, uniform(8) + gauss())
    assert run(tmp_path, "lock") == 0
    first = (tmp_path / "claims.lock.json").read_bytes()
    assert run(tmp_path, "lock", "--lock", str(tmp_path / "again.json")) == 0
    assert (tmp_path / "again.json").read_bytes() == first
    assert run(tmp_path, "lock") == 0
    assert (tmp_path / "claims.lock.json").read_bytes() == first


ENTRY = '"kind": "random", "fingerprint": "x", "environment": {}, "rtol": 0.0, "atol": 0.0'
MEAN = '{"hex": "0x1p0", "repr": "1.0"}'
SD = '{"hex": "0x1p-3", "repr": "0.125"}'


@pytest.mark.parametrize(
    "fields",
    [
        f'"value": {MEAN}, "sd": {SD}',
        f'"value": {MEAN}, "sd": {SD}, "replicates": 1',
        f'"value": {MEAN}, "sd": {SD}, "replicates": true',
        f'"value": {MEAN}, "sd": {SD}, "replicates": "8"',
        f'"value": {MEAN}, "sd": {SD}, "replicates": 8.0',
        f'"value": {MEAN}, "replicates": 8',
        f'"value": {MEAN}, "sd": "0.125", "replicates": 8',
        f'"value": {MEAN}, "sd": {{"hex": "-0x1p-3"}}, "replicates": 8',
        f'"value": {{"hex": "inf"}}, "sd": {SD}, "replicates": 8',
        f'"sd": {SD}, "replicates": 8',
    ],
)
def test_an_invalid_random_entry_is_a_configuration_error(tmp_path: Path, fields: str) -> None:
    text = f'{{"format": 1, "claims": {{"c": {{{ENTRY}, {fields}}}}}}}'
    assert json.loads(text)["claims"]["c"]["kind"] == "random"
    path = write(tmp_path, "claims.lock.json", text)
    with pytest.raises(ConfigError, match="lock file"):
        lockfile.read(path)


def test_verdicts_follow_the_holm_adjusted_p_values(tmp_path: Path) -> None:
    expected = gauss_summary()
    mean, se = expected.mean, expected.se
    far, near, low = (f"{mean + shift * se:.6f}" for shift in (8, 0.5, -8))
    readme = (
        sentinel(far) + sentinel(near) + sentinel(low, "lt") + sentinel(low, "gt") + sentinel("ten")
    )
    make_project(tmp_path, gauss(), {"README.md": readme})
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    assert status == 1
    assert report["claims"][0]["random"] == {
        "replicates": 32,
        "mean": mean,
        "sd": expected.sd,
        "se": se,
    }
    distances = [float(far) - 5e-7 - mean, float(near) - 5e-7 - mean, mean - float(low)]
    tails = [2 * stats.t.sf(distances[0] / se, 31), 2 * stats.t.sf(distances[1] / se, 31)]
    tails += [stats.t.sf(distances[2] / se, 31), 1.0]
    occurrences = report["occurrences"]
    assert [o["t"] for o in occurrences[:4]] == pytest.approx([d / se for d in distances] + [0.0])
    assert [o["p"] for o in occurrences[:4]] == pytest.approx(tails, rel=1e-9)
    assert [o["p_holm"] for o in occurrences[:4]] == pytest.approx(holm(tails), rel=1e-9)
    assert [(o["verdict"], o["result"]) for o in occurrences] == [
        ("violated", "fail"),
        ("supported", "pass"),
        ("violated", "fail"),
        ("supported", "pass"),
        (None, "bad-literal"),
    ]
    assert (occurrences[4]["t"], occurrences[4]["p"], occurrences[4]["p_holm"]) == (None,) * 3
    write(tmp_path, "README.md", sentinel(near) + sentinel(low, "gt"))
    status, report = json_report(tmp_path, "check")
    assert status == 0
    assert [o["verdict"] for o in report["occurrences"]] == ["supported", "supported"]


@pytest.mark.parametrize("option", ["atol", "rtol"])
def test_the_tolerance_of_a_random_claim_widens_eq_but_not_the_bounds(
    tmp_path: Path, option: str
) -> None:
    expected = gauss_summary()
    printed = f"{expected.mean + 8 * expected.se:.6f}"
    width = 9 * expected.se / (abs(expected.mean) if option == "rtol" else 1.0)
    files = {"README.md": sentinel(printed) + sentinel(printed, "gt")}
    make_project(tmp_path, gauss(f", {option}={width!r}"), files)
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    assert status == 1
    assert [(o["relation"], o["p"], o["verdict"]) for o in report["occurrences"]] == [
        ("eq", 1.0, "supported"),
        ("gt", pytest.approx(stats.t.sf(8.0, 31), rel=1e-4), "violated"),
    ]


def test_alpha_from_the_configuration_decides_a_borderline_number(tmp_path: Path) -> None:
    expected = gauss_summary()
    printed = f"{expected.mean + 2.45 * expected.se:.6f}"
    t = (float(printed) - 5e-7 - expected.mean) / expected.se
    p = 2 * float(stats.t.sf(t, 31))
    assert 0.01 < p < 0.05
    make_project(tmp_path, gauss(), {"README.md": sentinel(printed)})
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    occurrence = report["occurrences"][0]
    assert (status, report["alpha"], occurrence["verdict"]) == (0, 0.01, "supported")
    assert occurrence["p_holm"] == occurrence["p"] == pytest.approx(p, rel=1e-9)
    write(tmp_path, "numeric-claim.toml", DEFAULT_CONFIG + "alpha = 0.05\n")
    status, report = json_report(tmp_path, "check")
    occurrence = report["occurrences"][0]
    assert (status, report["alpha"], occurrence["verdict"]) == (1, 0.05, "violated")


def test_alpha_defaults_to_one_in_a_hundred_and_may_be_any_number_between_0_and_1(
    tmp_path: Path,
) -> None:
    make_project(tmp_path, "")
    assert load_config(tmp_path / "numeric-claim.toml").alpha == 0.01
    for value in ("1e-6", "0.5", "0.999"):
        write(tmp_path, "numeric-claim.toml", DEFAULT_CONFIG + f"alpha = {value}\n")
        assert load_config(tmp_path / "numeric-claim.toml").alpha == float(value)


@pytest.mark.parametrize(
    "value", ["0", "1", "0.0", "1.0", "-0.5", "1.5", '"0.05"', "true", "nan", "inf", "[0.05]"]
)
def test_an_alpha_outside_0_to_1_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    make_project(tmp_path, "", config=DEFAULT_CONFIG + f"alpha = {value}\n")
    assert run(tmp_path, "check") == 2
    assert "'alpha' must be a number between 0 and 1" in capsys.readouterr().err


def test_a_recomputed_random_claim_drifts_when_welchs_test_separates_the_means(
    tmp_path: Path,
) -> None:
    expected = gauss_summary()
    make_project(tmp_path, gauss())
    assert run(tmp_path, "lock") == 0

    def check_with_locked_mean(mean: float) -> dict[str, Any]:
        rewrite_lock(tmp_path, lambda claims: claims["g"].update(value=stored(mean)))
        status, report = json_report(tmp_path, "check", "--recompute")
        found: dict[str, Any] = report["claims"][0]
        assert status == (0 if found["status"] == "ok" else 1)
        return found

    assert check_with_locked_mean(expected.mean)["status"] == "ok"
    assert check_with_locked_mean(expected.mean + expected.se)["status"] == "ok"
    shifted = expected.mean + 6 * expected.se
    drifted = check_with_locked_mean(shifted)
    assert drifted["status"] == "drift"
    assert drifted["note"].startswith(
        f"computed mean {expected.mean!r} but locked {shifted!r} (Welch p="
    )
    p = stats.ttest_ind_from_stats(
        expected.mean, expected.sd, 32, shifted, expected.sd, 32, equal_var=False
    ).pvalue
    assert p < 1e-3


def test_without_spread_any_change_of_the_mean_is_drift(tmp_path: Path) -> None:
    make_project(tmp_path, logging_claims(CONSTANT))
    assert run(tmp_path, "lock") == 0
    assert lock_data(tmp_path)["c"]["sd"] == stored(0.0)
    status, report = json_report(tmp_path, "check", "--recompute")
    assert (status, report["claims"][0]["status"]) == (0, "ok")
    moved = math.nextafter(2.5, 3.0)
    rewrite_lock(tmp_path, lambda claims: claims["c"].update(value=stored(moved)))
    status, report = json_report(tmp_path, "check", "--recompute")
    assert (status, report["claims"][0]["status"]) == (1, "drift")


def test_a_random_claim_locked_as_another_kind_is_drift(tmp_path: Path) -> None:
    make_project(tmp_path, logging_claims(CONSTANT))
    assert run(tmp_path, "lock") == 0
    rewrite_lock(tmp_path, lambda claims: claims["c"].update(kind="seeded"))
    status, report = json_report(tmp_path, "check")
    found = report["claims"][0]
    assert (status, found["status"], found["source"]) == (1, "drift", "computed")
    assert found["note"] == "computed a random claim but locked a seeded one"


def test_the_text_report_shows_the_summary_and_each_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = uniform_values(8)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    se = sd / math.sqrt(8)
    high = f"{mean + 10 * se:.4f}"
    readme = sentinel("2", "lt", "u") + sentinel(high, "gt", "u")
    make_project(tmp_path, uniform(8), {"README.md": readme})
    assert run(tmp_path, "lock") == 0
    capsys.readouterr()
    status, report = json_report(tmp_path, "check")
    assert status == 1
    tested = report["occurrences"][1]
    assert tested["verdict"] == "violated"
    assert capsys.readouterr().out.splitlines() == [
        f"claim u [random] ok value={mean!r} (lock) mean={mean:.3g} sd={sd:.3g} se={se:.3g} R=8",
        f"README.md:1 u lt 2 value={mean!r} pass t=0 p=1 p_holm=1 verdict=supported",
        f"README.md:2 u gt {high} value={mean!r} fail t={tested['t']:.3g} p={tested['p']:.3g} "
        f"p_holm={tested['p_holm']:.3g} verdict=violated",
        "1 claims (1 ok), 2 occurrences (1 fail, 1 pass): exit 1",
    ]


def test_the_json_report_gives_other_kinds_no_random_data(tmp_path: Path) -> None:
    exact = '@claim("e")\ndef e():\n    return 3\n'
    readme = sentinel("3", name="e") + sentinel("2.5", name="c")
    make_project(tmp_path, logging_claims(exact, CONSTANT), {"README.md": readme})
    assert run(tmp_path, "lock") == 0
    status, report = json_report(tmp_path, "check")
    assert (status, report["alpha"]) == (0, 0.01)
    constant, other = report["claims"]
    assert constant["random"] == {"replicates": 8, "mean": 2.5, "sd": 0.0, "se": 0.0}
    assert other["random"] is None
    tests = {o["claim"]: (o["t"], o["p"], o["p_holm"], o["verdict"]) for o in report["occurrences"]}
    assert tests == {"e": (None, None, None, None), "c": (0.0, 1.0, 1.0, "supported")}


def test_list_shows_the_replicates_of_a_random_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = sentinel("1", "lt", "u")
    make_project(tmp_path, uniform(12, ', packages=["json"]'), {"README.md": readme})
    status, report = json_report(tmp_path, "list")
    assert (status, calls(tmp_path)) == (0, 0)
    assert capsys.readouterr().out.splitlines() == [
        "claim u [random] packages=json paths=- reference=- nudged=- ladders=- jitter=- "
        "replicates=12",
        "README.md:1 u lt 1 quantity=value",
        "1 claims, 1 occurrences",
    ]
    assert report["claims"] == [
        {
            "name": "u",
            "kind": "random",
            "packages": ["json"],
            "paths": [],
            "reference": None,
            "nudged": [],
            "ladders": {},
            "jitter": {},
            "replicates": 12,
        }
    ]


def test_the_package_tests_random_claims_without_numpy_scipy_or_mpmath() -> None:
    code = (
        "import sys\n"
        "for name in ('numpy', 'scipy', 'mpmath'):\n"
        "    sys.modules[name] = None\n"
        "import numeric_claim.cli\n"
        "from numeric_claim.montecarlo import t_sf\n"
        "print(repr(t_sf(1.5, 10)))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SOURCE), environment.get("PYTHONPATH")])
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout) == pytest.approx(float(stats.t.sf(1.5, 10)), rel=1e-9)
