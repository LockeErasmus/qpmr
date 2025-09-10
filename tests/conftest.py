"""
Pytest configuration
"""

import os
import sys

from _pytest.config.argparsing import Parser

repository_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_path = os.path.join(repository_path, "src")
#sys.path.append(repository_path)
sys.path.append(source_path)

def pytest_addoption(parser: Parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Enable plotting of test results"
    )

import pytest

@pytest.fixture
def enable_plot(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--plot")


# TODO logger via parser
from qpmr import init_logger
init_logger("DEBUG")