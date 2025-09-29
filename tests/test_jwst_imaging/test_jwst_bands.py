import os
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.instruments.jwst import JWST
from mejiro.galaxy_galaxy import SampleGG
from mejiro.synthetic_image import SyntheticImage
from mejiro.engines.stpsf_engine import STPSFEngine


@pytest.fixture
def test_data_dir():
    return os.path.join(os.path.dirname(mejiro.__path__[0]), 'tests', 'test_data')


@pytest.mark.parametrize("band", ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W'])
def test_band(band, test_data_dir):
    jwst = JWST()
    kwargs_psf = STPSFEngine.get_jwst_psf_kwargs(band, oversample=5, num_pix=101, check_cache=True, psf_cache_dir=test_data_dir, verbose=False)

    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                    instrument=jwst,
                                    band=band,
                                    kwargs_numerics={'supersampling_factor': 1},
                                    kwargs_psf=kwargs_psf,
                                    verbose=False)
        
    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        verbose=False)
