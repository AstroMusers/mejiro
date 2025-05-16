import numpy as np
import pytest
import matplotlib.pyplot as plt
from pyHalo.preset_models import preset_model_from_name

from mejiro.analysis import lensing
from mejiro.galaxy_galaxy import SampleBELLS, SampleSL2S, SampleGG


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_get_kappa(strong_lens):
    kappa = lensing.get_kappa(strong_lens.lens_model, strong_lens.kwargs_lens, scene_size=5, pixel_scale=0.11)
    
    assert kappa.shape == (47, 47)


def test_get_subhalo_mass_function():
    strong_lens = SampleGG()
    CDM = preset_model_from_name('CDM')
    realization = CDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), cone_opening_angle_arcsec=5)

    plt.loglog(*lensing.get_subhalo_mass_function(realization))
    plt.close()
