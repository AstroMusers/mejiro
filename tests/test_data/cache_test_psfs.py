import os

from mejiro.engines.stpsf_engine import STPSFEngine


def main():
    bands = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']

    here = os.path.dirname(os.path.abspath(__file__))

    for band in bands:
        psf_id = STPSFEngine.get_psf_id(band, 1, (2044, 2044), 1, 41)
        STPSFEngine.cache_psf(psf_id, '')

    # Also cache an F129 PSF at a second detector position so that the
    # test_roman_psf_varies_with_detector_position test can compare two
    # cached PSFs without having to run STPSF (which is slow).
    second_position_psf_id = STPSFEngine.get_psf_id('F129', 'SCA01', (200, 200), 5, 101)
    STPSFEngine.cache_psf(second_position_psf_id, here)


if __name__ == '__main__':
    main()
