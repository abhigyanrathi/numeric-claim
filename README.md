# numeric-claim

[![CI](https://github.com/abhigyanrathi/numeric-claim/actions/workflows/ci.yml/badge.svg)](https://github.com/abhigyanrathi/numeric-claim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/numeric-claim)](https://pypi.org/project/numeric-claim/)

Numbers printed in a README, a docstring or a paper go stale when the code behind them changes.
numeric-claim ties each printed number to the code that computes it, keeps the computed values in
a lock file committed with the code, and fails a check when a document and the code disagree. It
never rewrites a document: a failing check leaves the new wording to you.

It also asks whether a number deserves the digits it is printed with. A precision bound can be
smaller than the floating-point noise of the program that produced it, and a Monte Carlo result
printed from one seed can be a property of that seed rather than of the method. numeric-claim
measures both.

## Install

```
pip install numeric-claim
```

Python 3.12 or later, with no runtime dependencies. Install `numeric-claim[mpmath]` to give
claims a closed-form reference computed with mpmath.

## Quick start

Declare a claim in a Python module. Here `mypkg` stands for your own package:

```python
# claims.py
from numeric_claim import claim

from mypkg.pricing import call_price


@claim("atm_call", kind="seeded", packages=["mypkg"])
def atm_call() -> float:
    return call_price(spot=100.0, strike=100.0, rate=0.05, vol=0.2, maturity=1.0)
```

Point a configuration file at it:

```toml
# numeric-claim.toml
claims = ["claims.py"]
```

Mark the number where you print it. The marker is an HTML comment, so GitHub shows only the
number:

```markdown
# Pricing notes

A one-year at-the-money call is worth <!-- claim: atm_call -->10.45<!-- /claim -->.
```

Then compute the claims, commit `claims.lock.json`, and check:

```
$ numeric-claim lock
$ numeric-claim check
claim atm_call [seeded] ok value=10.450583572185565 (lock)
README.md:3 atm_call eq 10.45 value=10.450583572185565 pass
1 claims (1 ok), 1 occurrences (1 pass): exit 0
```

`10.45` passes because a printed number stands for everything that rounds to it, here anything
within 0.005 of 10.45. Print `10.46` and the check fails with exit status 1. Change the pricing
code and the check recomputes the claim, reports the drift from the locked value, and fails:

```
claim atm_call [seeded] drift value=10.2969358922752 (computed): computed 10.2969358922752 but locked 10.450583572185565
README.md:3 atm_call eq 10.45 value=10.2969358922752 fail
1 claims (1 drift), 1 occurrences (1 fail): exit 1
```

Run `numeric-claim check` in CI and stale numbers stop getting merged.

## Marking numbers

- A Markdown sentinel: `<!-- claim: NAME -->VALUE<!-- /claim -->`, or with a relation,
  `<!-- claim: NAME lt -->1e-8<!-- /claim -->`.
- A LaTeX macro: `\claim{NAME}{VALUE}` or `\claim[lt]{NAME}{VALUE}`, with
  `\newcommand{\claim}[3][]{#3}` in the preamble.
- A pattern in the configuration file, for text that cannot carry markup: an existing README, a
  comment in code, an assertion in a test. A pattern finds every match, including the ones you
  forgot about.

`eq`, the default relation, passes when the computed value lies within half a unit of the
printed number's last digit, widened by the claim's tolerance. `lt`, `le`, `gt` and `ge` read the
printed number as a bound.

The default scan reads every `.md` and `.tex` file under the project root, skipping `.git`,
virtual environments, build output and hidden directories. In Markdown, fenced code blocks and
inline code spans are skipped, so documentation can show the syntax without asserting it.

## Configuration

```toml
claims = ["claims.py"]                        # required: the modules that define claims
lock = "claims.lock.json"                     # where the lock file lives
distributions = ["numpy", "scipy", "mpmath"]  # versions recorded in every fingerprint
scan = ["**/*.md", "**/*.tex"]                # files searched for sentinels and macros
exclude = ["docs/drafts/**"]                  # files left out of that search
alpha = 0.01                                  # significance level for Monte Carlo claims

[[bind]]
files = ["README.md"]
pattern = '(\d[\d,]*) tests\b'                # one capture group: the printed number
claim = "test_count"                          # a claim name, or a list of names
relation = "eq"                               # eq, lt, le, gt or ge
```

Only `claims` is required. The other top-level values shown are the defaults, except `exclude`,
which is empty unless set.

## Claims

`claim(name, kind=..., ...)` marks a function whose value documents may quote. The function
takes no arguments, or one `ClaimContext` (`root`, `config_dir`, and the `seed` of a Monte Carlo
replicate), and returns an int or a float; numpy scalars are accepted. A claim lists the packages
and paths its value depends on in `packages` and `paths`, which decides when it is recomputed
(see [Lock file and fingerprints](#lock-file-and-fingerprints)).

There are three kinds:

- `exact`, the default: the value must reproduce exactly, like a count.
- `seeded`: a deterministic floating-point computation, compared with a tolerance (`rtol`,
  default 1e-12, and `atol`, default 0). A seeded claim can carry an error budget.
- `random`: a Monte Carlo computation, run once per replicate with its own seed. A printed number
  is read as a statement about the method's expected value.

### Error budgets

A precision claim such as "reproduced to <4e-8" means something only if the computation's own
rounding noise is well below 4e-8. To measure it, give the claim function its inputs and method
settings as keyword-only parameters with defaults, and name the value it reproduces:

```python
@claim(
    "t67_y15",
    kind="seeded",
    packages=["cosfin"],
    reference="50.279533980",
    ladders={"n": [1024, 2048], "l_trunc": [11.0, 12.0]},
    jitter={"l_trunc": 1e-9},
)
def t67_y15(
    *, spot: float = 100.0, strike: float = 100.0, n: int = 512, l_trunc: float = 10.0
) -> float: ...
```

numeric-claim then calls the function 32 times with each remaining nonzero float input nudged by
1 to 8 units in the last place, 32 times with the jittered settings moved by up to their relative
amplitude (between 1e-12 and 1e-6), and once for each ladder setting. The larger of the two
standard deviations is the rounding noise, sigma. Every printed number tied to the claim then
gets a verdict:

- `supported`: it holds with more than 3 sigma to spare;
- `rounding-limited`: it holds by less than 3 sigma, so rounding alone could break it on another
  machine;
- `violated`: it does not hold;
- `unsupported`: it bounds the error by no more than half a unit of the reference's last printed
  digit, which the reference itself cannot resolve.

To compare a printed number with the claim's error, `|value - reference|`, rather than its value,
bind it with `quantity = "error"`. The claim is also labelled with what dominates its error: the
ladder setting whose refinement reduces it most, `rounding`, `reference` (the error is within the
reference's last digit) or `unexplained`. A reference can be a closed form evaluated with mpmath,
for example `reference=closed_form(lambda mp: mp.pi, dps=50)`. The draws are seeded from the
claim's name, so the budget reproduces exactly on the same platform.

### Monte Carlo claims

```python
import numpy as np

from numeric_claim import ClaimContext, claim


@claim("hedge_ratio", kind="random", replicates=64, packages=["mypkg"])
def hedge_ratio(ctx: ClaimContext) -> float:
    rng = np.random.default_rng(ctx.seed)
    ...
```

Replicate `i` receives the first 8 bytes of the SHA-256 of `"hedge_ratio/i"` as its seed, so
every run reproduces. The claim's value is the mean of the replicates (32 unless `replicates`
says otherwise), with a standard error from their standard deviation. Each printed number gets a
Student t-test of what it says about the expected value: for `eq`, that the expected value lies
within half a unit of the printed number's last digit (two-sided); for a bound, that it lies on
the stated side (one-sided). The p-values of a run are adjusted together with Holm's procedure,
and a number whose adjusted p-value is at most `alpha` is `violated`. A number copied from one
seeded run usually carries more digits than the method supports, and fails.

A recomputed Monte Carlo claim drifts from its lock entry when Welch's t-test separates the two
means at `alpha`. The t distribution is computed inside the package, so this adds no dependency.

## Lock file and fingerprints

`numeric-claim lock` writes `claims.lock.json`: each claim's value, as exact hex next to a
readable form, with its fingerprint, the environment it was computed in, and its error budget or
Monte Carlo summary. The file is deterministic, so an unchanged run rewrites it byte for byte and
its git history shows when each number moved.

`check` reuses a locked value while the claim's fingerprint is unchanged and recomputes the claim
otherwise; `--recompute` forces it. The fingerprint covers the module that defines the claim;
every file in the claim's `packages` and `paths` (Python files by syntax tree, so comments and
formatting do not count but docstrings do); the Python version, operating system and machine; the
versions of the configured `distributions`; and numeric-claim's own version. A module that the
claims module imports is covered only if it is listed in `packages` or `paths`.

A recomputed seeded claim agrees with its lock entry within its tolerance, or within 3 sigma if it
has an error budget, so a lock written on Linux does not drift on Windows through rounding alone.

## Comparing revisions

```
$ numeric-claim diff HEAD~1
```

compares the lock file committed at a git revision (`HEAD` by default) with the current one, and
computes nothing. Each claim is `added`, `removed`, `unchanged`, `within-noise` or `moved`: an
exact claim moves when its value changes, a seeded claim when it changes by more than its
tolerance or 3 sigma, and a Monte Carlo claim when Welch's test separates the two means. The exit
status is 1 if any claim was added, removed or moved.

## Commands

| Command | What it does |
|---|---|
| `numeric-claim lock` | compute every claim and write the lock file |
| `numeric-claim check` | compare documents and the lock with the code; `--recompute` ignores the lock |
| `numeric-claim list` | show the claims and every marked number, computing nothing |
| `numeric-claim diff [REV]` | compare the lock at a git revision with the current one |

Every command takes `--config PATH` (default `./numeric-claim.toml`), `--root PATH` (default: the
configuration file's directory), `--lock PATH`, and `--json PATH` for a machine-readable report.
The exit status is 0 when everything agrees, 1 when a check fails and 2 for a configuration error.

## Tested on a real project

The test suite includes three gates that run numeric-claim on
[cosfin](https://github.com/abhigyanrathi/cosfin), a library of option-pricing methods, at
pinned commits. Each gate was written down before the code that meets it, together with cases the
tool must pass, so a tool that flagged everything would fail too. The configurations are in
[examples](https://github.com/abhigyanrathi/numeric-claim/tree/main/examples).

1. **A stale test count.** cosfin's README says "264 tests" in one place and "249 pytest" in
   another. Bound to the number of tests the suite actually collects, the first passes and the
   second fails.
2. **A precision claim below the noise.** cosfin states that two option prices from Table 6.7 of
   Oosterlee and Grzelak's *Mathematical Modeling and Computation in Finance* are reproduced "to
   <4e-8", and two from Table 6.8 "to <1e-8". For the Y = 1.5 case the rounding noise is about
   3.4e-8 on Linux. The error there is 3.47e-8: inside the bound but by less than 3 sigma, so
   `rounding-limited`. On Windows the same code gives 5.85e-8, `violated`. The other three
   bounds are `supported`, with their errors attributed to truncation, discretization and the
   reference's last digit.
3. **Monte Carlo results printed from one seed.** cosfin's README reports four delta-hedging
   results: P&L dispersion ratios of 3.82 and 1.93 between rebalancing schedules, a Merton-to-GBM
   dispersion ratio of 8.8 and a Merton mean P&L of -2.46. Each is a single run with seed 7.
   Declared as Monte Carlo claims over 64 replicates, the four have means of 3.857, 1.951, 8.622
   and -2.428, and all four printed numbers are flagged, 6.5 to 9.1 standard errors away. The
   six bounds that cosfin's own tests assert on the same experiments pass. Two lines of the
   report, from a Linux run:

```
README.md:117 ratio_13_208 eq 3.82 value=3.857354965780064 fail t=6.52 p=1.37e-08 p_holm=9.57e-08 verdict=violated
tests/test_ch5_hedging.py:25 ratio_13_208 gt 3.0 value=3.857354965780064 pass t=0 p=1 p_holm=1 verdict=supported
```

## Limitations

- The noise estimate is black-box: it measures how the output moves when the inputs and settings
  move slightly. A computation that rounds its own output coarsely, to float32 say, can show a
  sigma of 0, and its verdicts then reduce to a plain pass or fail.
- For exact and seeded claims an integer literal must match exactly, so an `eq` on an integer
  literal can never be `supported` on a claim with an error budget. For Monte Carlo claims the
  tolerance widens an integer literal like any other.
- Monte Carlo verdicts are statistical. Holm's adjustment keeps the chance of any false violation
  in a run at or below `alpha`, assuming the replicate mean is close to normally distributed, and
  the fixed seeds make whichever answer you get reproducible.
- On a Windows console the text report escapes characters the console cannot display; the
  `--json` report is exact.

## Development

```
git clone https://github.com/abhigyanrathi/numeric-claim
cd numeric-claim
pip install -e ".[dev]"
pytest
```

The gates need cosfin installed from the pinned commit and two checkouts of its repository:

```
pip install "cosfin @ git+https://github.com/abhigyanrathi/cosfin.git@39dee29614080f364f8c23c50f2b37279214f1ea"
git clone https://github.com/abhigyanrathi/cosfin.git ../cosfin-5f51ee5
git -C ../cosfin-5f51ee5 checkout --detach 5f51ee5135fd4cbd743e24d07d92969dde6fc4d8
git clone https://github.com/abhigyanrathi/cosfin.git ../cosfin-39dee29
git -C ../cosfin-39dee29 checkout --detach 39dee29614080f364f8c23c50f2b37279214f1ea
pytest tests/test_gate_claim_binding.py tests/test_gate_error_budget.py tests/test_gate_monte_carlo.py --cosfin-5f51ee5 ../cosfin-5f51ee5 --cosfin-39dee29 ../cosfin-39dee29
```

Put the test paths before the `--cosfin-...` options, as above. Gate 3 takes a few minutes.

The design decisions, each recorded before the code it governs, are in
[docs/decisions.md](https://github.com/abhigyanrathi/numeric-claim/blob/main/docs/decisions.md).

## License

MIT
