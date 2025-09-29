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


@pytest.mark.parametrize("strong_lens", [SampleGG()])
def test_jwst_imaging(strong_lens, test_data_dir):
    # TODO generate CDM realization with LOS and add to strong_lens

    jwst = JWST()
    kwargs_psf = STPSFEngine.get_jwst_psf_kwargs('F115W', oversample=5, num_pix=101, check_cache=True, psf_cache_dir=test_data_dir, verbose=False)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                    instrument=jwst,
                                    band='F115W',
                                    kwargs_numerics={'supersampling_factor': 1},
                                    kwargs_psf=kwargs_psf,
                                    verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        verbose=False)
