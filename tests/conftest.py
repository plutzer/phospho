"""Shared fixtures and pytest configuration.

The default run covers the fast unit tests plus the subset golden-master, which
reads the committed reference under ``tests/reference/``. Heavy tests over the
full (gitignored) ``ExampleData/`` are marked ``slow`` and run only with
``--runslow``.
"""

import os

import matplotlib
import pytest

matplotlib.use("Agg")

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")
SUBSET_INPUT = os.path.join(REFERENCE_DIR, "subset_input")
LEGACY_OUTPUT = os.path.join(REFERENCE_DIR, "legacy_output")
EXAMPLE_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ExampleData")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def subset_input():
    """Path to the committed golden-master subset input directory."""
    if not os.path.isdir(SUBSET_INPUT):
        pytest.skip("reference subset_input missing")
    return SUBSET_INPUT


@pytest.fixture(scope="session")
def legacy_output():
    """Path to the committed reference outputs from the original scripts."""
    if not os.path.isdir(LEGACY_OUTPUT):
        pytest.skip("reference legacy_output missing")
    return LEGACY_OUTPUT


@pytest.fixture(scope="session")
def example_data_dir():
    """Path to the full ExampleData directory; skip if absent (it is gitignored)."""
    if not os.path.isdir(EXAMPLE_DATA):
        pytest.skip("ExampleData/ not present")
    return EXAMPLE_DATA
