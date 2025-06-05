import pytest
from pyHalo.preset_models import preset_model_from_name

from mejiro.galaxy_galaxy import SampleBELLS, SampleSL2S, SampleGG
from mejiro.instruments.roman import Roman
from mejiro.synthetic_image import SyntheticImage


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_CDM(strong_lens):
    CDM = preset_model_from_name('CDM')
    realization = CDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), cone_opening_angle_arcsec=5)

    strong_lens.add_realization(realization)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False,
                                     verbose=False)
    

@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_single_halo(strong_lens):
    from pyHalo.single_realization import SingleHalo
    from pyHalo.Halos.lens_cosmo import LensCosmo
    from pyHalo.concentration_models import preset_concentration_models

    pyhalo_lens_cosmo = LensCosmo(strong_lens.z_lens, strong_lens.z_source)
    astropy_class = pyhalo_lens_cosmo.cosmo
    c_model, kwargs_concentration_model = preset_concentration_models('DIEMERJOYCE19')
    kwargs_concentration_model['scatter'] = False
    kwargs_concentration_model['cosmo'] = astropy_class
    concentration_model = c_model(**kwargs_concentration_model)
    truncation_model = None
    kwargs_halo_model = {
        'truncation_model': truncation_model,
        'concentration_model': concentration_model,
        'kwargs_density_profile': {}
    }
    single_halo = SingleHalo(halo_mass=10 ** 8,
                                x=1.12731457, y=-1.50967129,
                                mdef='NFW',
                                z=strong_lens.z_lens, zlens=strong_lens.z_lens, zsource=strong_lens.z_source,
                                subhalo_flag=True,
                                kwargs_halo_model=kwargs_halo_model,
                                astropy_instance=strong_lens.cosmo,
                                lens_cosmo=pyhalo_lens_cosmo)

    strong_lens.add_realization(single_halo)

    

@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_WDM(strong_lens):
    WDM = preset_model_from_name('WDM')
    realization = WDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), log_mc=7, cone_opening_angle_arcsec=5)

    strong_lens.add_realization(realization)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False,
                                     verbose=False)


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_SIDM(strong_lens):
    SIDM = preset_model_from_name('SIDM_core_collapse')
    mass_ranges_subhalos = [[6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0]]
    mass_ranges_field_halos = [[6.0, 7.5], [7.5, 8.5], [8.5, 10.0]]
    collapse_fraction_subhalos = [0.9, 0.7, 0.5, 0.2]
    collapse_fraction_fieldhalos = [0.3, 0.2, 0.1]
    realization = SIDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), mass_ranges_subhalos, mass_ranges_field_halos, collapse_fraction_subhalos, collapse_fraction_fieldhalos, cone_opening_angle_arcsec=5)

    strong_lens.add_realization(realization, use_jax=False)  # JAXtronomy doesn't support SPL_CORE profiles yet

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False,
                                     verbose=False)
    