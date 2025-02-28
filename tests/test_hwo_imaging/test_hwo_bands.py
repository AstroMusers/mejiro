import pytest

from mejiro.exposure import Exposure
from mejiro.instruments.hwo import HWO
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import lenstronomy_util


@pytest.mark.parametrize("band", ['B', 'FUV', 'H', 'I', 'J', 'K', 'NUV', 'R', 'U', 'V'])
def test_band(band):
    hwo = HWO()
    kwargs_psf = lenstronomy_util.get_gaussian_psf_kwargs(hwo.get_psf_fwhm(band))

    # The warning is expected: we're not testing for ray-shooting accuracy here, just that the code runs for different HWO bands, so we've set the oversampling factor to 1. This is not recommended for actual science cases.
    with pytest.warns(UserWarning,
                      match='Supersampling factor less than 5 may not be sufficient for accurate results, especially when convolving with a non-trivial PSF'):
        synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                        instrument=hwo,
                                        band=band,
                                        kwargs_numerics={'supersampling_factor': 1},
                                        verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        verbose=False)

    # TODO account for different pixel scales for different filters
    # assert synthetic_image.pixel_scale == 0.008
    # assert synthetic_image.native_pixel_scale == 0.04
    # assert synthetic_image.num_pix == 625
    # assert synthetic_image.native_num_pix == 125
    # assert synthetic_image.arcsec == 5.0
    # assert synthetic_image.image.shape == (625, 625)

    # TODO checks on the images
