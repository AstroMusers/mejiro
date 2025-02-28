import pytest

from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_default_roman_imaging(strong_lens):
    # TODO generate CDM realization with LOS and add to strong_lens
    # TODO grab a PSF from STPSF

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        verbose=False)
