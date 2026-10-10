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
