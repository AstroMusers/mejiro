import os
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage


TEST_DATA_DIR = os.path.join(os.path.dirname(mejiro.__path__[0]), 'tests', 'test_data')


@pytest.fixture
def roman_technical_information_path():
    return os.getenv("ROMAN_TECHNICAL_DOCUMENTATION_PATH", None)


@pytest.mark.parametrize("band", ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213'])
def test_band(band, roman_technical_information_path):
    roman = Roman(roman_technical_information_path)
    lens = SampleStrongLens()
    scene_size = 5  # arcsec
    oversample = 5
    exposure_time = 146
    instrument_params = {
        'detector': 1,
        'detector_position': (2048, 2048)
    }

    synthetic_image = SyntheticImage(strong_lens=lens,
                                     instrument=roman,
                                     band=band,
                                     arcsec=scene_size,
                                     oversample=oversample,
                                     instrument_params=instrument_params,
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=True)

    # TODO checks on the images
