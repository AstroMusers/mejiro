import matplotlib
matplotlib.use('Agg')

import multiprocessing
import os

import pytest

import mejiro


def pytest_configure(config):
    # pytest runs as a multi-threaded process; forking from it risks deadlocks in
    # child workers (Python 3.12+ DeprecationWarning). forkserver pre-starts a
    # single-threaded server so all subprocess forks originate from a clean context.
    multiprocessing.set_start_method('forkserver', force=True)


@pytest.fixture(scope='session')
def test_data_dir():
    """Absolute path to the tests/test_data directory (PSF caches, pickles, etc.)."""
    return os.path.join(os.path.dirname(os.path.dirname(mejiro.__file__)), 'tests', 'test_data')


@pytest.fixture
def sample_gg():
    """Fresh SampleGG instance per test. Function-scoped because tests routinely
    mutate `physical_params` and `kwargs_lens` to exercise specific code paths."""
    from mejiro.galaxy_galaxy import SampleGG
    return SampleGG()


@pytest.fixture
def roman_instrument():
    """Fresh Roman instance per test. Function-scoped because some methods
    (e.g., `get_zeropoint_magnitude`) lazily populate instance attributes."""
    from mejiro.instruments.roman import Roman
    return Roman()
