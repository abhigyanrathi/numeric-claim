# Design decisions

Decisions for numeric-claim, recorded before the code they govern. Entries are
append-only: a later change is a new entry, never an edit.

## 1. Verify published numbers; never regenerate them

numeric-claim checks the numbers a project publishes (in a README, docs,
docstrings or LaTeX) against values computed from the project's own code, and
fails when they disagree. It never rewrites a document. Tools such as Cog
regenerate text from code on every run, which does not work for computations
that are slow or random, and silently rewriting a published number hides the
fact that it moved. A failing check leaves the new wording to a person.

## 2. Name

The distribution is `numeric-claim` and the import package `numeric_claim`.
The earlier working name `attest` is taken on PyPI. `numeric-claim` and
`numclaim` were both unregistered on PyPI when checked on 10 October 2026;
being unregistered does not reserve a name.

## 3. One committed lock file

Locked values live in one `claims.lock.json`, committed to git, so its history
records how each claim moved and every change shows up in an ordinary diff.
The file is written deterministically (sorted keys, fixed indentation, floats
stored as exact hex next to a readable form): an unchanged run produces an
identical file.

## 4. Fingerprint boundary: the package edge

A locked value is reused without recomputation only while its claim's
fingerprint is unchanged. The fingerprint covers the module that defines the
claim; every file in the packages and paths the claim declares, Python files
by syntax tree (comments and formatting do not count, code and docstrings do)
and other files by their bytes; the Python version and implementation; the
operating system and machine; and the installed versions of a configurable
list of distributions (numpy, scipy and mpmath by default). Tracing the exact
call graph would recompute less, but static tracing misses dynamic calls in
Python. Hashing whole packages never misses a change; the cost is extra
recomputation. The operating system is included because the same code can
give different floating-point results on different platforms.

## 5. Tying document numbers to claims

A number in a document is tied to a claim in one of three ways:
- a Markdown sentinel, `<!-- claim: NAME -->VALUE<!-- /claim -->`, which
  GitHub renders invisibly;
- a LaTeX macro, `\claim{NAME}{VALUE}`, defined in the preamble as
  `\newcommand{\claim}[3][]{#3}`;
- a pattern in the configuration file: a list of files and a regular
  expression with one capture group, applied to every match.
Patterns serve text that cannot or should not carry markup, such as an
existing README or a string in source code, and they find every occurrence,
including ones the author has forgotten. A sentinel or macro may name a
relation; a pattern declares one. With `eq`, a printed number agrees with a
computed one when they differ by at most half a unit in the last printed
digit plus the claim's own tolerance; `lt`, `le`, `gt` and `ge` treat the
printed number as a bound.

## 6. Error-budget probes treat the code as a black box

To judge a precision claim, numeric-claim calls the claim's computation with
different settings instead of instrumenting the code: ladders over
discretization and truncation settings, tiny jitter of continuous method
settings that leaves the exact answer unchanged, and random nudges of a few
units in the last place to every floating-point input. The spread of results
under jitter and nudges estimates the rounding noise. mpmath is used only to
evaluate closed-form reference values at high precision. This works on code
that cannot be edited.

## 7. Multiple comparisons: Holm

Random (Monte Carlo) claims are tested statistically, several in one run. To
hold the chance of any false failure across a run at the chosen level,
p-values are adjusted with Holm's step-down procedure, which controls the same
family-wise error rate as Bonferroni under any dependence and never rejects
less.

## 8. Validation gates

The tool must catch two real problems in the cosfin repository
(https://github.com/abhigyanrathi/cosfin). Each gate is pinned to a commit so
later fixes to cosfin cannot remove it.

Gate 1, claim binding, at cosfin commit
5f51ee5135fd4cbd743e24d07d92969dde6fc4d8. The README says "264 tests" on line
32 and "249 pytest" on line 94; the suite collects 264 tests at that commit.
With the test count tied by pattern to both phrasings, `numeric-claim check`
must fail line 94, pass line 32 and exit with status 1.

Gate 2, error budget, at cosfin commit
39dee29614080f364f8c23c50f2b37279214f1ea. `src/cosfin/replications.py`
states that the Table 6.7 references (Y=0.5 and Y=1.5) are "reproduced here
to <4e-8" and that the Table 6.8 references (T=0.1 and T=1) are "reproduced
to <1e-8 at the book's N=4096". From its own probes, without being told the
outcome, numeric-claim must:
- flag the Y=1.5 bound, as rounding-limited where the measured error is
  nominally inside it (Linux: about 3.5e-8) and as violated where it is not
  (cosfin's docs/RESULTS.md records 5.85e-8 on Windows);
- pass the Y=0.5 bound and both Table 6.8 bounds.
Exploratory probes run before the build (Linux, CPython 3.13, numpy 2.5.3)
show why. Under jitter and nudges the Y=1.5 call price has a standard
deviation of about 2.8e-8, more than five times the 5.3e-9 margin: the direct
call sum cancels terms whose absolute values add up to about 8e6. Priced as a
put plus put-call parity, the same case lands within 1.2e-10 of the
reference. The Y=0.5 and Table 6.8 errors (1.8e-9, 2.0e-10 and 8.5e-9) move
by less than 1e-11 under the same probes, so they are method or reference
error, not rounding. The Table 6.8 T=0.1 bound has only a 1.5e-9 margin, but
its error is deterministic, so it must pass; a rule based on the margin alone
would fail it.

## 9. Error budgets

A seeded claim can carry an error budget, which estimates the rounding noise
in its value and shows how the value moves with its method settings. The
claim function takes its inputs and settings as keyword-only parameters with
numeric defaults, and the defaults give the claim's value. A claim may also
name a reference, the value it reproduces: a printed number, whose last digit
gives its precision, or a closed form evaluated with mpmath (50 significant
digits by default) and treated as exact.

Whenever numeric-claim computes a budgeted claim, it also calls it 32 times
with every nonzero floating-point input moved by 1 to 8 units in the last
place in a random direction (nudges), 32 times with each declared continuous
setting scaled by 1 + a*u, for a declared relative amplitude a of at most
1e-6 and u uniform on [-1, 1] (jitter), and once at each declared refinement
of a setting (ladders). The noise sigma is the larger of the standard
deviations over the nudged and the jittered calls. The draws come from a
generator seeded with the claim's name, so the lock file stays byte-identical
across runs; it stores the two standard deviations and the ladder values
beside the value, and `check` reuses them while the fingerprint is unchanged.

Every number tied to a budgeted claim gets a verdict. The number is compared
with the claim's value or, when its binding asks for it, with the value's
error against the reference. The margin is how far inside the relation the
compared quantity lies: the bound minus the quantity for `lt` and `le`, the
quantity minus the bound for `gt` and `ge`, and the allowed difference minus
the actual one for `eq`. A failing relation is `violated`; one that holds
with a margin of at most 3 sigma is `rounding-limited`, because rounding on
another platform could move the quantity across the edge; any other is
`supported`. An upper bound on the error that is no larger than the
reference's half-unit is `unsupported`, because the reference cannot resolve
it. Only `supported` passes. A margin alone cannot separate a fragile pass
from a safe one; the measured noise can.

When a budgeted claim has a reference, numeric-claim also names what
dominates its error e, taking the first that applies: the setting whose
ladder reduces e the most, if it reduces e by more than 3 sigma and by at
least half of e; `rounding`, when 3 sigma is at least half of e;
`reference`, when e is within the reference's half-unit; otherwise
`unexplained`. The label never changes the exit status. Growth of the error
with a setting is not read as rounding: at a fixed number of terms a wider
truncation range also coarsens the discretization, so a deterministic error
can grow with it too.

## 10. Claim values may be numpy scalars

A claim may return any integer or real number type, such as numpy's int64 or
float32, not only Python's int and float. Integers are converted to int and
other reals to float before they are checked and stored; booleans, numpy's
included, are still rejected. Numerical code returns numpy scalars as a
matter of course, and a cast in every claim would add clutter without making
anything safer.

## 11. Gate 2 in detail

Gate 2's claims are four calls to cosfin's `cos_european_price`, one per
case. Every model and contract parameter is an input, apart from the zero
dividend yield. The number of terms N and the truncation width L are
settings: L is jittered by a relative 1e-9, and the ladders are N in
{1024, 2048} for Table 6.7 (base 512) or {8192, 16384} for Table 6.8 (base
4096), and L in {11, 12} (base 10). The references are written as
replications.py writes them, to nine decimals; docs/RESULTS.md shows the
Table 6.7 ones to eight because its script prints them as floats, and
neither reading changes a verdict or label below. Patterns tie each of the
two bounds to the errors of its two cases.

Beyond the verdicts of entry 8, the gate requires these labels: `l_trunc`
for Y=0.5 (L=11 removes most of its 1.8e-9 error), `rounding` for Y=1.5, `n`
for T=0.1 (N=8192 brings its 8.5e-9 error down to 1.7e-9) and `reference`
for T=1 (its 2.0e-10 error is within the reference's half-unit of 5e-10).
Y=1.5 counts as flagged when it is `rounding-limited` with its error below
4e-8, or `violated` with its error at or above 4e-8.

The constants of the rule (32 draws, 3 sigma, half of the error) were chosen
after the exploratory probes of entry 8, so the gate is not a blind test of
them. Re-measured on 11 October 2026 (Linux, CPython 3.12.3, numpy 2.5.4)
with the draws this design specifies, Y=1.5 has a sigma of 3.4e-8 against a
margin of 5.3e-9, and each of the other three margins exceeds 4,900 sigma:
any multiple of sigma from 0.16 to 4,900 gives the same verdicts (0.25 to
3,500 across 20 other seeds).

Entry 8 says the Y=0.5 and Table 6.8 errors move by less than 1e-11 under
the probes. That holds for the Table 6.8 errors, which move by about 1e-13,
but not for Y=0.5: its standard deviation under the probes is up to about
1e-11, and the spread of its draws up to about 5e-11. The conclusion stands:
these errors are method or reference error, not rounding.

## 12. Monte Carlo claims

A random claim's value comes from a Monte Carlo method. numeric-claim runs the
claim R times (32 by default, set per claim), passing replicate i a seed
derived from the claim's name and i, and reads a printed number as a statement
about the method's expected value. The mean of the replicates estimates that
value, with standard error s/sqrt(R) from their standard deviation s. For `eq`,
the expected value must lie within half a unit of the printed number's last
digit, widened by the claim's tolerance; `lt`, `le`, `gt` and `ge` read the
printed number as a bound. Each printed number gets a Student t-test: t is the
distance from the mean to the nearest value at which the statement holds,
divided by the standard error, with R - 1 degrees of freedom, two-sided for
`eq` and one-sided for the bounds. The p-values of a run are adjusted together
with Holm's procedure (entry 7); a printed number whose adjusted p-value is at
most alpha (0.01 unless the configuration sets it) is `violated`, any other is
`supported`. One run with a fixed seed reproduces exactly and stays a seeded
claim; a random claim states what the method gives on average, so digits finer
than that average supports fail.

Seeding the replicates from the claim's name makes a check repeatable: a result
does not flicker between runs, and the lock file, which stores R, the mean and
the standard deviation, stays byte-identical. A recomputed random claim drifts
from its lock entry when Welch's two-sample t-test separates the two means at
alpha. The t distribution is computed in the package, from the regularized
incomplete beta function, so numeric-claim keeps no runtime dependencies.

## 13. Comparing locks across revisions

`numeric-claim diff REV` compares the lock file committed at a git revision
(HEAD by default) with the current one, and computes nothing. Each claim is
added, removed, unchanged, within noise or moved. An exact claim moves when its
value changes; a seeded claim when its value changes by more than its
tolerance, or by more than 3 sigma if it has an error budget; a random claim
when Welch's test separates the two means at alpha. The command exits with
status 1 when a claim was added, removed or moved. This is differential testing
over time: a claim that moved between two revisions shows up even when no
document quotes it, and the history of the lock file says when it moved.

## 14. Scanning skips code

The default scan reads every Markdown and LaTeX file, and documentation that
explains the claim syntax quotes it: this repository's own decisions.md does,
and failed its own check. In Markdown, sentinels and macros inside fenced code
blocks and inline code spans are now ignored, since code there is shown, not
asserted. An `exclude` list of globs removes files from the scan. Patterns in
`[[bind]]` are unchanged: they name their files and read them whole, code
included.

## 15. Error budgets, second pass

Five changes after the first run of the error budget. The fingerprint now
includes numeric-claim's own version, so a lock written under other probe or
test rules is recomputed rather than reused. A jitter amplitude must be at
least 1e-12: below about 1e-16, 1 + a*u rounds to 1, the setting never moves,
and the noise would read as zero. Ladder parameters may not be named
`rounding`, `reference` or `unexplained`, which would read as labels. The
drift of a budgeted claim tolerates 3 sigma (the larger of the locked and the
recomputed sigma), so a lock committed on one platform does not drift on
another through rounding alone: the Table 6.7 Y=1.5 value differs by 2.4e-8
between Linux and Windows, against a sigma of about 3e-8. Fingerprints skip
`*.egg-info` directories, which an editable install writes into the source
tree.

## 16. Gate 3: Monte Carlo claims

At cosfin commit 39dee29614080f364f8c23c50f2b37279214f1ea, README.md lines
117 to 119 report four delta-hedging results: dispersion ratios of 3.82 (13
against 208 rebalancing steps) and 1.93 (13 against 52), a Merton-to-GBM
dispersion ratio of 8.8 at 208 steps, and a Merton mean P&L of -2.46. Each is
a single run with seed 7 (reproduced exactly: 3.8235, 1.9252, 8.7883 and
-2.4635). As random claims over the same experiments (20,000 paths each, 64
replicates), numeric-claim must flag all four as `violated`, and must pass the
six bounds that cosfin's tests/test_ch5_hedging.py asserts on lines 25, 26, 39
and 42 (3.0 < ratio < 5.0, 1.6 < ratio < 2.4, Merton dispersion above 4.0
times GBM's, mean P&L below -1.0), so a tool that flags everything cannot
pass.

Probes run on 15 October 2026 (Linux, CPython 3.12.3, numpy 2.5.4), before the
build and with the seeds this design specifies, put the four expected values at
3.857, 1.951, 8.622 and -2.428. Each lies outside its printed number's
half-unit by at least 6.5 standard errors at 64 replicates, while the six
bounds hold by more than 150. The number of replicates and alpha were chosen
after these probes.
