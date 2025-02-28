import os
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.galaxy_galaxy import SampleGG
from mejiro.synthetic_image import SyntheticImage
from mejiro.engines.stpsf_engine import STPSFEngine


@pytest.fixture
def test_data_dir():
    return os.path.join(os.path.dirname(mejiro.__path__[0]), 'tests', 'test_data')


@pytest.mark.parametrize("band", ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213'])
def test_band(band, test_data_dir):
    kwargs_psf = STPSFEngine.get_psf_kwargs(band, 'SCA01', (2048, 2048), oversample=5, num_pix=101, check_cache=True, psf_cache_dir=test_data_dir, verbose=False)

    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band=band,
                                     fov_arcsec=5,
                                     instrument_params={'detector': 1, 'detector_position': (2048, 2048)},
                                     kwargs_psf=kwargs_psf,
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        verbose=False)

    # TODO checks on the images
