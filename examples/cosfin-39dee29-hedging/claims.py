"""Monte Carlo claims for the delta-hedging results of the cosfin checkout at commit 39dee29."""

import numpy
import numpy.typing as npt
from cosfin.pricing.hedging import delta_hedge_pnl

from numeric_claim import ClaimContext, claim

ARGS = (100.0, 100.0, 0.05, 0.2, 1.0)
PATHS = 20_000
MERTON = {"lam": 1.0, "mu_j": -0.1, "sigma_j": 0.15}


def pnl(ctx: ClaimContext, n_steps: int, **jumps: float) -> npt.NDArray[numpy.float64]:
    """Hedging P&L per path, from a new generator seeded with the replicate's seed."""
    rng = numpy.random.default_rng(ctx.seed)
    return delta_hedge_pnl(*ARGS, n_steps, PATHS, rng=rng, **jumps)


@claim("ratio_13_208", kind="random", replicates=64, packages=["cosfin"])
def ratio_13_208(ctx: ClaimContext) -> float:
    """GBM P&L dispersion with 13 rebalancing steps over that with 208."""
    return float(pnl(ctx, 13).std() / pnl(ctx, 208).std())


@claim("ratio_13_52", kind="random", replicates=64, packages=["cosfin"])
def ratio_13_52(ctx: ClaimContext) -> float:
    """GBM P&L dispersion with 13 rebalancing steps over that with 52."""
    return float(pnl(ctx, 13).std() / pnl(ctx, 52).std())


@claim("merton_gbm_208", kind="random", replicates=64, packages=["cosfin"])
def merton_gbm_208(ctx: ClaimContext) -> float:
    """P&L dispersion with 208 steps under Merton jumps over that under GBM."""
    return float(pnl(ctx, 208, **MERTON).std() / pnl(ctx, 208).std())


@claim("merton_mean_208", kind="random", replicates=64, packages=["cosfin"])
def merton_mean_208(ctx: ClaimContext) -> float:
    """Mean P&L with 208 steps under Merton jumps."""
    return float(pnl(ctx, 208, **MERTON).mean())
