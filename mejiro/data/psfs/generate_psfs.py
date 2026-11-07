"""Generate the PSFs shipped with mejiro, in the ``roman/`` and ``jwst/`` directories here.

``PipelineHelper`` uses these directories as the PSF cache when ``psf_cache_dir`` is ``null``.
This script needs the ``roman`` optional dependency and STPSF's data files:

    python mejiro/data/psfs/generate_psfs.py

It overwrites the existing PSFs, so that they all come from the same version of STPSF. The PSFs
the test suite uses are separate: see ``tests/test_data/cache_test_psfs.py``.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mejiro.engines.stpsf_engine import STPSFEngine

HERE = os.path.dirname(os.path.abspath(__file__))

ROMAN_BANDS = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']
ROMAN_DETECTOR = 1
ROMAN_DETECTOR_POSITION = (2043, 2043)  # the detector center, as divide_up_sca(1) computes it

JWST_BANDS = ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']

OVERSAMPLE = 1
NUM_PIX = 41


def save_roman_psf(band):
    psf_id = STPSFEngine.get_psf_id(band, ROMAN_DETECTOR, ROMAN_DETECTOR_POSITION, OVERSAMPLE, NUM_PIX)
    psf = STPSFEngine.get_roman_psf(band, ROMAN_DETECTOR, ROMAN_DETECTOR_POSITION, OVERSAMPLE, NUM_PIX)
    np.save(os.path.join(HERE, 'roman', f'{psf_id}.npy'), psf)


def save_jwst_psf(band):
    psf_id = STPSFEngine.get_jwst_psf_id(band, OVERSAMPLE, NUM_PIX)
    psf = STPSFEngine.get_jwst_psf(band, OVERSAMPLE, NUM_PIX)
    np.save(os.path.join(HERE, 'jwst', f'{psf_id}.npy'), psf)


def main():
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(save_roman_psf, band) for band in ROMAN_BANDS]
        futures += [executor.submit(save_jwst_psf, band) for band in JWST_BANDS]
        for future in futures:
            future.result()  # raise any error from the workers


if __name__ == '__main__':
    main()
