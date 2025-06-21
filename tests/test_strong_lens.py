import numpy as np
import pytest

from pyHalo.preset_models import preset_model_from_name
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_get_kappa(strong_lens):
    kappa = strong_lens.get_kappa()
    assert kappa is not None

@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_get_realization_kappa(strong_lens):
    with pytest.raises(ValueError):
        realization_kappa = strong_lens.get_realization_kappa()

    CDM = preset_model_from_name('CDM')
    realization = CDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), cone_opening_angle_arcsec=5, log_m_host=np.log10(strong_lens.get_main_halo_mass()))

    strong_lens.add_realization(realization)
    realization_kappa = strong_lens.get_realization_kappa()
    assert realization_kappa is not None
