"""Step 00 has to use the same PSF cache as the steps after it.

``Pipeline(_test_mode=True)`` skips step 00, so ``test_pipeline_run`` can't catch a mismatch.
"""
import argparse
import os

import yaml

import mejiro
from mejiro.pipeline import _00_cache_psfs

SIMPLE_CONFIG = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'simple.yaml')


def _fail_if_computing(*args, **kwargs):
    raise AssertionError('step 00 tried to compute PSFs')


def test_finds_cached_psfs_with_a_fresh_data_dir(tmp_path, monkeypatch):
    # simple.yaml's relative psf_cache_dir falls back to the repository's tests/test_data, which
    # holds all of its PSFs. Looking under data_dir instead would miss them, and creating that
    # directory would also turn off the fallback for the later steps.
    monkeypatch.setattr(_00_cache_psfs, 'ProcessPoolExecutor', _fail_if_computing)

    _00_cache_psfs.main(argparse.Namespace(config=SIMPLE_CONFIG, data_dir=str(tmp_path)))

    assert not (tmp_path / 'tests').exists(), 'step 00 created a PSF cache under data_dir'


def test_null_psf_cache_dir(tmp_path):
    # null means the default cache in the package. No bands, so there's nothing to compute.
    with open(SIMPLE_CONFIG) as f:
        config = yaml.safe_load(f)
    config['psf_cache_dir'] = None
    config['psf']['bands'] = []
    config_file = tmp_path / 'null_psf_cache_dir.yaml'
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

    _00_cache_psfs.main(argparse.Namespace(config=str(config_file), data_dir=str(tmp_path)))
