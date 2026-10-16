import math
import random

import pytest
from scipy import stats

from numeric_claim.literals import parse_literal
from numeric_claim.montecarlo import Summary, holm, t_sf, t_test, welch

DEGREES = [1, 2, 3, 5, 7.5, 10, 31, 63, 100, 1000, 100000]
GRID = [step / 10 for step in range(401)]
SUMMARY = Summary(10.0, 2.0, 25)


@pytest.mark.parametrize("df", DEGREES)
def test_the_t_tail_agrees_with_scipy_to_one_part_in_a_billion(df: float) -> None:
    compared = 0
    for t in GRID:
        expected = float(stats.t.sf(t, df))
        if expected > 1e-300:
            assert t_sf(t, df) == pytest.approx(expected, rel=1e-9, abs=0), t
            compared += 1
    assert compared >= 370


@pytest.mark.parametrize("df", [1, 7.5, 63, 100000])
def test_a_negative_t_takes_the_complement(df: float) -> None:
    for t in (0.1, 0.5, 2.0, 10.0):
        assert t_sf(-t, df) == 1.0 - t_sf(t, df)
        assert t_sf(-t, df) == pytest.approx(float(stats.t.sf(-t, df)), rel=1e-9)


def test_the_t_tail_at_zero_and_beyond_any_float_square() -> None:
    assert t_sf(0.0, 10) == 0.5
    assert t_sf(math.inf, 10) == 0.0
    assert t_sf(-math.inf, 10) == 1.0
    assert t_sf(1e200, 10) == 0.0


@pytest.mark.parametrize(
    ("relation", "printed", "tolerance", "distance"),
    [
        ("eq", "11.0", 0.0, 0.95),
        ("eq", "10.1", 0.0, 0.05),
        ("eq", "9.2", 0.0, 0.75),
        ("eq", "9", 0.0, 1.0),
        ("eq", "9", 0.25, 0.75),
        ("eq", "11.0", 0.5, 0.45),
        ("lt", "9.5", 0.0, 0.5),
        ("le", "9.5", 0.0, 0.5),
        ("lt", "9.5", 0.25, 0.5),
        ("gt", "10.8", 0.0, 0.8),
        ("ge", "10.8", 0.0, 0.8),
        ("ge", "10.8", 0.25, 0.8),
    ],
)
def test_the_p_value_of_each_relation_family_against_scipy(
    relation: str, printed: str, tolerance: float, distance: float
) -> None:
    t, p = t_test(relation, parse_literal(printed), SUMMARY, tolerance)
    assert t == pytest.approx(distance / 0.4, rel=1e-12)
    tail = float(stats.t.sf(distance / 0.4, 24))
    assert p == pytest.approx(2 * tail if relation == "eq" else tail, rel=1e-9)


@pytest.mark.parametrize(
    ("relation", "printed", "tolerance"),
    [
        ("eq", "10.0", 0.0),
        ("eq", "10", 0.0),
        ("eq", "9.7", 0.25),
        ("eq", "10.3", 0.25),
        ("lt", "10", 0.0),
        ("le", "10", 0.0),
        ("lt", "12", 0.0),
        ("gt", "10", 0.0),
        ("ge", "9.5", 0.0),
    ],
)
def test_a_statement_the_mean_satisfies_has_a_p_value_of_one(
    relation: str, printed: str, tolerance: float
) -> None:
    assert t_test(relation, parse_literal(printed), SUMMARY, tolerance) == (0.0, 1.0)


def test_without_a_standard_error_the_p_value_is_one_or_zero() -> None:
    constant = Summary(10.0, 0.0, 16)
    assert constant.se == 0
    assert t_test("eq", parse_literal("10.0"), constant, 0.0) == (0.0, 1.0)
    assert t_test("eq", parse_literal("9.9"), constant, 0.05) == (0.0, 1.0)
    assert t_test("lt", parse_literal("10"), constant, 0.0) == (0.0, 1.0)
    assert t_test("ge", parse_literal("10"), constant, 0.0) == (0.0, 1.0)
    assert t_test("eq", parse_literal("10.2"), constant, 0.0) == (math.inf, 0.0)
    assert t_test("le", parse_literal("9.99"), constant, 0.0) == (math.inf, 0.0)
    assert t_test("gt", parse_literal("10.5"), constant, 0.0) == (math.inf, 0.0)


def test_an_unknown_relation_is_rejected() -> None:
    with pytest.raises(ValueError, match="relation"):
        t_test("ne", parse_literal("1"), SUMMARY, 0.0)


def test_holm_adjusts_the_example_of_the_specification() -> None:
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02], rel=1e-12)


def test_holm_gives_tied_p_values_the_same_adjustment() -> None:
    assert holm([0.02, 0.01, 0.02, 0.01]) == pytest.approx([0.04, 0.04, 0.04, 0.04], rel=1e-12)
    assert holm([0.3, 0.001, 0.3]) == pytest.approx([0.6, 0.003, 0.6], rel=1e-12)


def test_holm_caps_at_one_and_leaves_a_single_p_value_alone() -> None:
    assert holm([]) == []
    assert holm([0.2]) == [0.2]
    assert holm([0.5, 0.6, 0.9]) == [1.0, 1.0, 1.0]


def test_holm_adjusted_values_are_monotone_in_the_raw_ones() -> None:
    rng = random.Random(7)
    raw = [rng.random() ** 4 for _ in range(50)]
    adjusted = holm(raw)
    ranked = [adjusted[index] for index in sorted(range(len(raw)), key=raw.__getitem__)]
    assert ranked == sorted(ranked)
    assert all(p <= q <= 1.0 for p, q in zip(raw, adjusted, strict=True))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (Summary(1.0, 0.5, 32), Summary(1.3, 0.8, 20)),
        (Summary(-2.0, 3.0, 8), Summary(-2.1, 0.1, 100000)),
        (Summary(5.0, 1.0, 64), Summary(5.0, 1.0, 64)),
    ],
)
def test_welch_against_scipy(first: Summary, second: Summary) -> None:
    expected = stats.ttest_ind_from_stats(
        first.mean,
        first.sd,
        first.replicates,
        second.mean,
        second.sd,
        second.replicates,
        equal_var=False,
    ).pvalue
    assert welch(first, second) == pytest.approx(float(expected), rel=1e-9)
    assert welch(second, first) == welch(first, second)


def test_welch_without_any_spread_compares_the_means() -> None:
    assert welch(Summary(1.0, 0.0, 8), Summary(1.0, 0.0, 16)) == 1.0
    assert welch(Summary(1.0, 0.0, 8), Summary(1.5, 0.0, 16)) == 0.0
    assert welch(Summary(1.0, 0.0, 8), Summary(1.0, 0.25, 16)) == 1.0
