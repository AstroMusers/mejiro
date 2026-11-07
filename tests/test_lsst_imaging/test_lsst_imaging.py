import numpy as np
import pytest

from mejiro.exposure import Exposure
from mejiro.galaxy_galaxy import SampleGG
from mejiro.instruments.lsst import LSST
from mejiro.synthetic_image import SyntheticImage


def sample_lens(bands):
    # SampleGG has no LSST magnitudes, so give it some
    lens = SampleGG()
    for band in bands:
        lens.physical_params['magnitudes']['lens'][band] = 19
        lens.physical_params['magnitudes']['source'][band] = 21
    return lens


@pytest.mark.parametrize('band', LSST().bands)
def test_lsst_exposure(band):
    lsst = LSST()
    synthetic_image = SyntheticImage(strong_lens=sample_lens(lsst.bands),
                                     instrument=lsst,
                                     band=band,
                                     fov_arcsec=10,
                                     kwargs_numerics={'supersampling_factor': 1},
                                     kwargs_psf=lsst.get_psf_kwargs(band),
                                     pieces=True)
    assert synthetic_image.pixel_scale == pytest.approx(0.2)
    assert np.sum(synthetic_image.data) > 0

    exposure = Exposure(synthetic_image, exposure_time=6900, engine='galsim', engine_params={'rng_seed': 42})
    assert exposure.data.shape == synthetic_image.data.shape
    assert np.all(np.isfinite(exposure.data))
    for noise in (exposure.sky_background, exposure.poisson_noise, exposure.dark_noise, exposure.read_noise):
        assert noise is not None
