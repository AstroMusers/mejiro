import pytest

from mejiro.exposure import Exposure
from mejiro.instruments.hwo import HWO
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import lenstronomy_util


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_hwo_imaging(strong_lens):
    # TODO generate CDM realization with LOS and add to strong_lens
    
    hwo = HWO()
    kwargs_psf = lenstronomy_util.get_gaussian_psf_kwargs(hwo.get_psf_fwhm('J'))

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                    instrument=hwo,
                                    band='J',
                                    kwargs_numerics={'supersampling_factor': 1},
                                    kwargs_psf=kwargs_psf,
                                    verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        verbose=False)
