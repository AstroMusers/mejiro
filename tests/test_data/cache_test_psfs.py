"""Generate the PSFs the test suite reads from this directory.

CI has no STPSF data files, so a test fails there if a PSF it needs isn't cached here. When a
test needs a new PSF, add it to the lists below and run this script, which needs the ``roman``
optional dependency and STPSF's data files:

    python tests/test_data/cache_test_psfs.py

It overwrites the existing PSFs, so that they all come from the same version of STPSF.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.utils.roman_util import divide_up_sca

HERE = os.path.dirname(os.path.abspath(__file__))

ROMAN_BANDS = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']
JWST_BANDS = ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']

# (band, detector, detector position, oversample, num_pix)
ROMAN_PSFS = [
    # tests that build their own PSF, e.g. test_flux.py and test_roman_bands.py
    *[(band, 1, (2043, 2043), 5, 101) for band in ROMAN_BANDS],
    # test_roman_psf_varies_with_detector_position compares this with the F129 PSF above
    ('F129', 1, (200, 200), 5, 101),
    # the psf section of simple.yaml, which test.yaml's is a subset of
    *[(band, detector, (2043, 2043), 1, 73) for band in ['F106', 'F158'] for detector in [1, 18]],
    # the psf section of roman_test.yaml
    *[(band, detector, position, 1, 101)
      for band in ['F106', 'F129', 'F158']
      for detector in [1, 18]
      for position in divide_up_sca(2)],
]

# (band, oversample, num_pix)
JWST_PSFS = [(band, 5, 101) for band in JWST_BANDS]


def save_roman_psf(band, detector, detector_position, oversample, num_pix):
    psf_id = STPSFEngine.get_psf_id(band, detector, detector_position, oversample, num_pix)
    psf = STPSFEngine.get_roman_psf(band, detector, detector_position, oversample, num_pix)
    np.save(os.path.join(HERE, f'{psf_id}.npy'), psf)


def save_jwst_psf(band, oversample, num_pix):
    psf_id = STPSFEngine.get_jwst_psf_id(band, oversample, num_pix)
    psf = STPSFEngine.get_jwst_psf(band, oversample, num_pix)
    np.save(os.path.join(HERE, f'{psf_id}.npy'), psf)


def main():
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(save_roman_psf, *psf) for psf in ROMAN_PSFS]
        futures += [executor.submit(save_jwst_psf, *psf) for psf in JWST_PSFS]
        for future in futures:
            future.result()  # raise any error from the workers


if __name__ == '__main__':
    main()
