import matplotlib
matplotlib.use('Agg')

import multiprocessing


def pytest_configure(config):
    # pytest runs as a multi-threaded process; forking from it risks deadlocks in
    # child workers (Python 3.12+ DeprecationWarning). forkserver pre-starts a
    # single-threaded server so all subprocess forks originate from a clean context.
    multiprocessing.set_start_method('forkserver', force=True)
