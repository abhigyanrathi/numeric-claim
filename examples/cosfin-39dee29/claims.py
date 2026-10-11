"""Claims for the cosfin checkout at commit 39dee29."""

from cosfin.models.characteristic_functions import CGMY, VarianceGamma
from cosfin.numerical.cos import cos_european_price

from numeric_claim import claim

JITTER = {"l_trunc": 1e-9}
TABLE_6_7_LADDERS = {"n": [1024, 2048], "l_trunc": [11.0, 12.0]}
TABLE_6_8_LADDERS = {"n": [8192, 16384], "l_trunc": [11.0, 12.0]}


@claim(
    "t67_y05",
    kind="seeded",
    packages=["cosfin"],
    reference="21.679593920",
    ladders=TABLE_6_7_LADDERS,
    jitter=JITTER,
)
def t67_y05(
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    r: float = 0.1,
    t: float = 1.0,
    c: float = 1.0,
    g: float = 5.0,
    m: float = 5.0,
    y: float = 0.5,
    sigma: float = 0.2,
    n: int = 512,
    l_trunc: float = 10.0,
) -> float:
    """COS price of the Table 6.7 CGMYB call with Y = 0.5."""
    model = CGMY(r=r, q=0.0, C=c, G=g, M=m, Y=y, sigma=sigma)
    price = cos_european_price(model.cf, model.cumulants, spot, strike, r, t, n, l_trunc=l_trunc)
    return float(price[0])


@claim(
    "t67_y15",
    kind="seeded",
    packages=["cosfin"],
    reference="50.279533980",
    ladders=TABLE_6_7_LADDERS,
    jitter=JITTER,
)
def t67_y15(
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    r: float = 0.1,
    t: float = 1.0,
    c: float = 1.0,
    g: float = 5.0,
    m: float = 5.0,
    y: float = 1.5,
    sigma: float = 0.2,
    n: int = 512,
    l_trunc: float = 10.0,
) -> float:
    """COS price of the Table 6.7 CGMYB call with Y = 1.5."""
    model = CGMY(r=r, q=0.0, C=c, G=g, M=m, Y=y, sigma=sigma)
    price = cos_european_price(model.cf, model.cumulants, spot, strike, r, t, n, l_trunc=l_trunc)
    return float(price[0])


@claim(
    "t68_t01",
    kind="seeded",
    packages=["cosfin"],
    reference="10.993703187",
    ladders=TABLE_6_8_LADDERS,
    jitter=JITTER,
)
def t68_t01(
    *,
    spot: float = 100.0,
    strike: float = 90.0,
    r: float = 0.1,
    t: float = 0.1,
    sigma: float = 0.12,
    theta: float = -0.14,
    beta: float = 0.2,
    n: int = 4096,
    l_trunc: float = 10.0,
) -> float:
    """COS price of the Table 6.8 Variance Gamma call with T = 0.1."""
    model = VarianceGamma(r=r, q=0.0, sigma=sigma, theta=theta, beta=beta)
    price = cos_european_price(model.cf, model.cumulants, spot, strike, r, t, n, l_trunc=l_trunc)
    return float(price[0])


@claim(
    "t68_t1",
    kind="seeded",
    packages=["cosfin"],
    reference="19.099354724",
    ladders=TABLE_6_8_LADDERS,
    jitter=JITTER,
)
def t68_t1(
    *,
    spot: float = 100.0,
    strike: float = 90.0,
    r: float = 0.1,
    t: float = 1.0,
    sigma: float = 0.12,
    theta: float = -0.14,
    beta: float = 0.2,
    n: int = 4096,
    l_trunc: float = 10.0,
) -> float:
    """COS price of the Table 6.8 Variance Gamma call with T = 1."""
    model = VarianceGamma(r=r, q=0.0, sigma=sigma, theta=theta, beta=beta)
    price = cos_european_price(model.cf, model.cumulants, spot, strike, r, t, n, l_trunc=l_trunc)
    return float(price[0])
