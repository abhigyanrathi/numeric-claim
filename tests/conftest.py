import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cosfin-5f51ee5",
        action="store",
        default=None,
        metavar="PATH",
        help="checkout of cosfin at 5f51ee5135fd4cbd743e24d07d92969dde6fc4d8; "
        "enables the claim-binding gate",
    )
    parser.addoption(
        "--cosfin-39dee29",
        action="store",
        default=None,
        metavar="PATH",
        help="checkout of cosfin at 39dee29614080f364f8c23c50f2b37279214f1ea; "
        "enables the error-budget and Monte Carlo gates",
    )
