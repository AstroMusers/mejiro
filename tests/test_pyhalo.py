import pytest
import numpy as np
from pyHalo.preset_models import preset_model_from_name

from mejiro.galaxy_galaxy import SampleBELLS, SampleSL2S, SampleGG
from mejiro.instruments.roman import Roman
from mejiro.synthetic_image import SyntheticImage


def _circular_aperture_mask(shape, center_xy, radius):
    ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _assert_realization_attached(strong_lens, realization):
    """Assertions that should hold after `strong_lens.add_realization(realization)`."""
    # macromodel must be preserved
    assert strong_lens.realization is realization
    assert hasattr(strong_lens, 'kwargs_lens_macromodel')
    assert hasattr(strong_lens, 'lens_model_list_macromodel')

    # exact-count check: add_realization should append exactly the entries
    # that lensing_quantities() returns, without dropping or duplicating any.
    expected_added = len(realization.lensing_quantities(add_mass_sheet_correction=True)[0])
    actual_added = len(strong_lens.kwargs_lens) - len(strong_lens.kwargs_lens_macromodel)
    assert actual_added == expected_added, (
        f"add_realization appended {actual_added} entries to kwargs_lens but "
        f"realization.lensing_quantities() reports {expected_added}"
    )

    # kwargs_lens, lens_model_list, lens_redshift_list must grow in lockstep
    assert (
        len(strong_lens.kwargs_lens)
        == len(strong_lens.lens_model_list)
        == len(strong_lens.lens_redshift_list)
    )


def _assert_synthetic_image_data_sane(synthetic_image):
    data = synthetic_image.data
    assert isinstance(data, np.ndarray)
    assert data.shape == (synthetic_image.num_pix, synthetic_image.num_pix)
    assert np.all(np.isfinite(data))
    assert np.sum(data) > 0


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_CDM(strong_lens):
    CDM = preset_model_from_name('CDM')
    realization = CDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), cone_opening_angle_arcsec=5, log_m_host=np.log10(strong_lens.get_main_halo_mass()))

    strong_lens.add_realization(realization)
    _assert_realization_attached(strong_lens, realization)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False)
    _assert_synthetic_image_data_sane(synthetic_image)


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
    _assert_realization_attached(strong_lens, single_halo)


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_WDM(strong_lens):
    WDM = preset_model_from_name('WDM')
    realization = WDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), log_mc=7, cone_opening_angle_arcsec=5, log_m_host=np.log10(strong_lens.get_main_halo_mass()))

    strong_lens.add_realization(realization)
    _assert_realization_attached(strong_lens, realization)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False)
    _assert_synthetic_image_data_sane(synthetic_image)


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_SIDM(strong_lens):
    SIDM = preset_model_from_name('SIDM_core_collapse')
    mass_ranges_subhalos = [[6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0]]
    mass_ranges_field_halos = [[6.0, 7.5], [7.5, 8.5], [8.5, 10.0]]
    collapse_fraction_subhalos = [0.9, 0.7, 0.5, 0.2]
    collapse_fraction_fieldhalos = [0.3, 0.2, 0.1]
    realization = SIDM(round(strong_lens.z_lens, 2), 
                       round(strong_lens.z_source, 2), 
                       mass_ranges_subhalos, 
                       mass_ranges_field_halos, 
                       collapse_fraction_subhalos,
                       collapse_fraction_fieldhalos, 
                       cone_opening_angle_arcsec=5, 
                       x_core_halo=0.1,
                       log_m_host=np.log10(strong_lens.get_main_halo_mass())
                       )

    strong_lens.add_realization(realization, use_jax=False)  # JAXtronomy doesn't support SPL_CORE profiles yet
    _assert_realization_attached(strong_lens, realization)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False)
    _assert_synthetic_image_data_sane(synthetic_image)


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_ULDM(strong_lens):
    ULDM = preset_model_from_name('ULDM')
    realization = ULDM(round(strong_lens.z_lens, 2), 
                       round(strong_lens.z_source, 2), 
                       log10_m_uldm=-22,
                       cone_opening_angle_arcsec=5, 
                       log_m_host=np.log10(strong_lens.get_main_halo_mass()),
                       flucs_shape='ring',
                       flucs_args={'angle': 0.0, 'rmin': 0.9, 'rmax': 1.1},
                       log10_fluc_amplitude=-1.6, 
                       n_cut=1000000)

    strong_lens.add_realization(realization)
    _assert_realization_attached(strong_lens, realization)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     pieces=False)
    _assert_synthetic_image_data_sane(synthetic_image)


def test_substructure_changes_image():
    """Differential check: rendering the same lens with vs. without
    a CDM realization should produce a measurably different image,
    both globally and in apertures around the lensed-image positions.
    """
    common = dict(
        instrument=Roman(),
        band='F129',
        fov_arcsec=5,
        instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
        kwargs_numerics={},
        kwargs_psf={},
        pieces=False,
    )

    # macromodel-only image
    sl_smooth = SampleGG()
    img_smooth = SyntheticImage(strong_lens=sl_smooth, **common)

    # same lens + a CDM realization
    sl_clumpy = SampleGG()
    CDM = preset_model_from_name('CDM')
    realization = CDM(
        round(sl_clumpy.z_lens, 2),
        round(sl_clumpy.z_source, 2),
        cone_opening_angle_arcsec=5,
        log_m_host=np.log10(sl_clumpy.get_main_halo_mass()),
    )
    sl_clumpy.add_realization(realization)
    img_clumpy = SyntheticImage(strong_lens=sl_clumpy, **common)

    # global differential check
    assert img_smooth.data.shape == img_clumpy.data.shape
    diff = img_clumpy.data - img_smooth.data
    assert np.max(np.abs(diff)) > 0, "substructure produced an identical image"
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(img_smooth.data)
    assert rel_l2 > 1e-4, (
        f"substructure barely perturbs the image (rel L2 = {rel_l2:.2e}); "
        "either the realization was empty or it failed to propagate into the lens model"
    )

    # aperture-level differential check at the predicted image positions
    px, py = img_smooth.get_image_positions(pixel=True)
    aperture_radius = 3
    smooth_aperture_fluxes = []
    clumpy_aperture_fluxes = []
    for cx, cy in zip(px, py):
        if not (0 <= cx < img_smooth.data.shape[1] and 0 <= cy < img_smooth.data.shape[0]):
            continue
        mask = _circular_aperture_mask(img_smooth.data.shape, (cx, cy), aperture_radius)
        smooth_aperture_fluxes.append(img_smooth.data[mask].sum())
        clumpy_aperture_fluxes.append(img_clumpy.data[mask].sum())

    assert len(smooth_aperture_fluxes) >= 1
    assert all(f > 0 for f in smooth_aperture_fluxes)
    assert all(f > 0 for f in clumpy_aperture_fluxes)

    rel_changes = [
        abs(c - s) / s
        for c, s in zip(clumpy_aperture_fluxes, smooth_aperture_fluxes)
    ]
    assert max(rel_changes) > 1e-4, (
        f"substructure should perturb at least one aperture; rel_changes={rel_changes}"
    )
