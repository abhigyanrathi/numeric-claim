import math
import re
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy
import pytest

from helpers import make_project, run
from numeric_claim import claim, closed_form
from numeric_claim.claim import ClaimSpec, ClosedForm, Probes, spec_of
from numeric_claim.literals import Literal


def area() -> Callable[..., float]:
    def function(
        *,
        x: float = 2.0,
        y: float = -3.0,
        zero: float = 0.0,
        count: int = 5,
        n: int = 10,
        width: float = 10.0,
    ) -> float:
        return x * y + zero + count + n + width

    return function


def declared(function: Callable[..., Any], **options: Any) -> ClaimSpec:
    spec = spec_of(claim("c", kind="seeded", **options)(function))
    assert spec is not None
    return spec


def test_a_budgeted_claim_declares_its_reference_and_what_its_probes_vary() -> None:
    function = area()
    spec = declared(
        function,
        reference="-6.00",
        ladders={"width": [11, 12.5], "n": (20, 40)},
        jitter={"width": 1e-9},
    )
    assert spec.reference == Literal("-6.00", Decimal("-6.00"), Decimal("0.005"), False)
    assert spec.probes == Probes(
        defaults={"x": 2.0, "y": -3.0, "zero": 0.0, "count": 5, "n": 10, "width": 10.0},
        nudged=("x", "y"),
        jitter={"width": 1e-9},
        ladders={"n": (20, 40), "width": (11, 12.5)},
    )
    assert function() == 19.0


def test_ladders_and_jitter_are_kept_in_signature_order() -> None:
    spec = declared(
        area(),
        ladders={"width": [11.0], "n": [20], "x": [3.0]},
        jitter={"width": 1e-9, "y": 1e-7},
    )
    assert spec.probes is not None
    assert list(spec.probes.ladders) == ["x", "n", "width"]
    assert list(spec.probes.jitter) == ["y", "width"]
    assert spec.probes.nudged == ()


def test_ladder_settings_stay_as_declared() -> None:
    spec = declared(area(), ladders={"width": [11, numpy.float64(12.5)], "n": range(20, 41, 20)})
    assert spec.probes is not None
    assert [type(setting) for setting in spec.probes.ladders["width"]] == [int, float]
    assert spec.probes.ladders == {"n": (20, 40), "width": (11, 12.5)}


def test_a_parameter_may_have_both_a_ladder_and_a_jitter() -> None:
    spec = declared(lambda *, width=10.0: width, ladders={"width": [11.0]}, jitter={"width": 1e-6})
    assert spec.probes is not None
    assert spec.probes.ladders == {"width": (11.0,)}
    assert spec.probes.jitter == {"width": 1e-6}


def test_a_seeded_claim_without_keyword_only_parameters_has_no_budget() -> None:
    assert declared(lambda: 1.0).probes is None
    assert declared(lambda ctx: 1.0, reference="1.0", ladders={}, jitter={}).probes is None


def test_a_reference_is_a_number_literal_or_a_closed_form() -> None:
    literal = declared(lambda: 1.0, reference=" 21.679593920 ").reference
    assert isinstance(literal, Literal)
    assert (literal.text, literal.half_unit) == ("21.679593920", Decimal("5e-10"))
    form = closed_form(lambda mp: mp.pi)
    assert declared(lambda: 1.0, reference=form).reference is form
    assert isinstance(form, ClosedForm)
    assert declared(lambda: 1.0).reference is None


INT64 = numpy.int64(3)
INVALID: list[tuple[dict[str, Any], Callable[..., Any], str]] = [
    ({"kind": "exact", "reference": "1.0"}, lambda: 1, "an exact claim takes no reference"),
    ({"kind": "exact", "ladders": {}}, lambda: 1, "an exact claim takes no reference"),
    ({"kind": "exact", "jitter": {}}, lambda: 1, "an exact claim takes no reference"),
    ({"kind": "exact"}, lambda *, x=1.0: x, "an exact claim takes no keyword-only parameters"),
    ({"reference": "about 3"}, lambda: 1.0, "invalid reference: not a number: 'about 3'"),
    ({"reference": ""}, lambda: 1.0, "invalid reference"),
    ({"reference": "1e99999999"}, lambda: 1.0, "invalid reference: number out of range"),
    ({"reference": 1.5}, lambda: 1.0, "reference must be a number written as a string"),
    ({}, lambda *, x: x, "keyword-only parameter 'x' needs a default"),
    ({}, lambda *, x=1.0, flag=True: x, "the default of 'flag' must be an int or a finite float"),
    ({}, lambda *, x=1.0, mode="fast": x, "the default of 'mode' must be"),
    ({}, lambda *, x=1.0, other=None: x, "the default of 'other' must be"),
    ({}, lambda *, x=math.nan: x, "the default of 'x' must be"),
    ({}, lambda *, x=math.inf: x, "the default of 'x' must be"),
    ({}, lambda *, x=Fraction(1, 2): x, "the default of 'x' must be"),
    ({}, lambda *, x=INT64, y=1.0: x, "the default of 'x' must be"),
    ({"ladders": {"missing": [1]}}, lambda *, x=1.0: x, "ladders names 'missing'"),
    ({"jitter": {"missing": 1e-9}}, lambda *, x=1.0: x, "jitter names 'missing'"),
    ({"ladders": {"ctx": [1]}}, lambda ctx, *, x=1.0: x, "ladders names 'ctx'"),
    ({"ladders": {"n": [20]}}, lambda: 1.0, "ladders names 'n'"),
    ({"ladders": [("n", [20])]}, lambda *, x=1.0, n=10: x, "ladders must map"),
    ({"jitter": 1e-9}, lambda *, x=1.0: x, "jitter must map"),
    ({"ladders": {"n": []}}, lambda *, x=1.0, n=10: x, "the ladder for 'n' is empty"),
    ({"ladders": {"n": 20}}, lambda *, x=1.0, n=10: x, "must be a sequence of settings"),
    ({"ladders": {"n": [20.0]}}, lambda *, x=1.0, n=10: x, "must be an int, got 20.0"),
    ({"ladders": {"n": [True]}}, lambda *, x=1.0, n=10: x, "must be an int, got True"),
    ({"ladders": {"w": ["11"]}}, lambda *, x=1.0, w=10.0: x, "an int or a finite float"),
    ({"ladders": {"w": [True]}}, lambda *, x=1.0, w=10.0: x, "an int or a finite float"),
    ({"ladders": {"w": [math.inf]}}, lambda *, x=1.0, w=10.0: x, "an int or a finite float"),
    ({"ladders": {"w": [11.0, math.nan]}}, lambda *, x=1.0, w=10.0: x, "a finite float"),
    ({"jitter": {"w": 0}}, lambda *, w=10.0: w, "must be in (0, 1e-06], got 0"),
    ({"jitter": {"w": 0.0}}, lambda *, w=10.0: w, "must be in (0, 1e-06]"),
    ({"jitter": {"w": -1e-9}}, lambda *, w=10.0: w, "must be in (0, 1e-06]"),
    ({"jitter": {"w": 1.0000001e-6}}, lambda *, w=10.0: w, "must be in (0, 1e-06]"),
    ({"jitter": {"w": math.nan}}, lambda *, w=10.0: w, "must be in (0, 1e-06]"),
    ({"jitter": {"w": "1e-9"}}, lambda *, w=10.0: w, "must be in (0, 1e-06]"),
    ({"jitter": {"w": True}}, lambda *, w=10.0: w, "must be in (0, 1e-06]"),
    ({"jitter": {"n": 1e-9}}, lambda *, x=1.0, n=10: x, "jitter needs a float default"),
    ({}, lambda *, n=10: n, "needs a nonzero float input to nudge or a jitter"),
    ({}, lambda *, zero=0.0, negative_zero=-0.0: zero, "needs a nonzero float input"),
    ({"ladders": {"x": [2.0]}}, lambda *, x=1.0, n=10: x, "needs a nonzero float input"),
]


@pytest.mark.parametrize(("options", "function", "message"), INVALID)
def test_an_invalid_budget_raises_when_the_decorator_runs(
    options: dict[str, Any], function: Callable[..., Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        claim("c", **{"kind": "seeded", **options})(function)


@pytest.mark.parametrize("amplitude", [1e-6, 1e-9, 5e-324])
def test_a_jitter_amplitude_up_to_one_in_a_million_is_accepted(amplitude: float) -> None:
    spec = declared(lambda *, w=10.0: w, jitter={"w": amplitude})
    assert spec.probes is not None
    assert spec.probes.jitter == {"w": amplitude}


def test_an_invalid_budget_in_a_claims_module_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = (
        "from numeric_claim import claim\n\n\n"
        "@claim('c', kind='seeded', jitter={'n': 1e-9})\ndef c(*, x=1.0, n=10):\n    return x\n"
    )
    make_project(tmp_path, claims)
    assert run(tmp_path, "check") == 2
    error = capsys.readouterr().err
    assert "error while loading claims module" in error
    assert "ValueError: jitter needs a float default" in error
