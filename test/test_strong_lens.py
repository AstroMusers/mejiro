from astropy.cosmology import default_cosmology
from pytest import approx

from mejiro.lenses import strong_lens
from mejiro.lenses.strong_lens import StrongLens
from mejiro.lenses.test import SampleStrongLens


def test_init():
    kwargs_model = {'cosmo': default_cosmology.get(),
 'lens_light_model_list': ['SERSIC_ELLIPSE'],
 'lens_model_list': ['SIE', 'SHEAR', 'CONVERGENCE'],
 'lens_redshift_list': [0.2902115249535011,
                        0.2902115249535011,
                        0.2902115249535011],
 'source_light_model_list': ['SERSIC_ELLIPSE'],
 'source_redshift_list': [0.5876899931818929],
 'z_source': 0.5876899931818929,
 'z_source_convention': 5}
    
    kwargs_params = {'kwargs_lens': [{'center_x': -0.007876281728887604,
            'center_y': 0.010633393703246008,
            'e1': 0.004858808997848661,
            'e2': 0.0075210751726143355,
            'theta_E': 1.168082477232392},
            {'dec_0': 0,
            'gamma1': -0.03648819840013156,
            'gamma2': -0.06511863424492038,
            'ra_0': 0},
            {'dec_0': 0, 'kappa': 0.06020941823541971, 'ra_0': 0}],
'kwargs_lens_light': [{'R_sersic': 0.5300707454127908,
                'center_x': -0.007876281728887604,
                'center_y': 0.010633393703246008,
                'e1': 0.023377277902774978,
                'e2': 0.05349948216860632,
                'magnitude': 17.5664222662219,
                'n_sersic': 4.0}],
'kwargs_ps': None,
'kwargs_source': [{'R_sersic': 0.1651633078964498,
            'center_x': 0.30298310338567075,
            'center_y': -0.3505004565139597,
            'e1': -0.06350855238708408,
            'e2': -0.08420760408362458,
            'magnitude': 21.434711611915137,
            'n_sersic': 1.0}]}

    lens_mags = {'F106': 17.5664222662219,
'F129': 17.269983557132853,
'F184': 17.00761457389914}

    source_mags = {'F106': 21.434711611915137,
'F129': 21.121205893763328,
'F184': 20.542431041034558}
    
    lens_stellar_mass = 286796906929.3925
    lens_vel_disp = 295.97270864848
    snr = 10.
    uid = 42
    
    lens = StrongLens(kwargs_model=kwargs_model, kwargs_params=kwargs_params, lens_mags=lens_mags, source_mags=source_mags, lens_stellar_mass=lens_stellar_mass, lens_vel_disp=lens_vel_disp,
                 snr=snr, uid=uid)

    assert lens.kwargs_model == kwargs_model
    assert lens.kwargs_params == kwargs_params
    assert lens.lens_mags == lens_mags
    assert lens.source_mags == source_mags
    assert lens.lens_stellar_mass == lens_stellar_mass
    assert lens.lens_vel_disp == lens_vel_disp
    assert lens.snr == snr
    assert lens.uid == uid

    assert lens.z_lens == 0.2902115249535011
    assert lens.z_source == 0.5876899931818929

# def test_get_lenstronomy_kwargs():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     band = 'F106'
#     lenstronomy_kwargs = lens.get_lenstronomy_kwargs(band)
#     assert lenstronomy_kwargs == {'kwargs_model': {}, 'kwargs_params': {}, 'kwargs_special': {'kwargs_lens_light': {}, 'kwargs_source': {}}}

# def test_get_macrolens_kappa():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     num_pix = 100
#     cone = 1.0
#     kappa = lens.get_macrolens_kappa(num_pix, cone)
#     assert kappa == 0.0  # Replace with expected value

# def test_get_kappa():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     num_pix = 100
#     subhalo_cone = 1.0
#     _get_kappa_macro = False
#     kappa = lens.get_kappa(num_pix, subhalo_cone, _get_kappa_macro)
#     assert kappa == 0.0  # Replace with expected value

# def test_get_delta_kappa():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     num_pix = 100
#     subhalo_cone = 1.0
#     delta_kappa = lens.get_delta_kappa(num_pix, subhalo_cone)
#     assert delta_kappa == 0.0  # Replace with expected value

# def test_get_einstein_radius():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     einstein_radius = lens.get_einstein_radius()
#     assert einstein_radius == 0.0  # Replace with expected value

# def test_get_main_halo_mass():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     main_halo_mass = lens.get_main_halo_mass()
#     assert main_halo_mass == 0.0  # Replace with expected value

# def test_mass_in_einstein_radius():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     mass_in_einstein_radius = lens.mass_in_einstein_radius()
#     assert mass_in_einstein_radius == 0.0  # Replace with expected value

# def test_generate_cdm_subhalos():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     log_mlow = 6
#     log_mhigh = 10
#     subhalo_cone = 10
#     los_normalization = 0
#     r_tidal = 0.5
#     sigma_sub = 0.055
#     cdm_subhalos = lens.generate_cdm_subhalos(log_mlow, log_mhigh, subhalo_cone, los_normalization, r_tidal, sigma_sub)
#     assert cdm_subhalos == []  # Replace with expected value

# def test_add_subhalos():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     realization = 1
#     return_stats = False
#     suppress_output = True
#     lens.add_subhalos(realization, return_stats, suppress_output)
#     # Add assertions based on the expected behavior

# def test_get_lens_flux_cps():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     band = 'F106'
#     lens_flux_cps = lens.get_lens_flux_cps(band)
#     assert lens_flux_cps == 0.0  # Replace with expected value

# def test_get_source_flux_cps():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     band = 'F106'
#     source_flux_cps = lens.get_source_flux_cps(band)
#     assert source_flux_cps == 0.0  # Replace with expected value

# def test_get_total_flux_cps():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     band = 'F106'
#     total_flux_cps = lens.get_total_flux_cps(band)
#     assert total_flux_cps == 0.0  # Replace with expected value

def test_get_array():
    sample_lens = SampleStrongLens()

    num_pix = 45
    side = 4.95
    band = 'F184'
    kwargs_psf = {'psf_type': 'NONE'}

    array = sample_lens.get_array(num_pix, side, band, kwargs_psf)

    assert array.shape == (num_pix, num_pix)

# def test_get_amp_light_kwargs():
#     magnitude_zero_point = 25.0
#     light_model_class = 'SERSIC_ELLIPSE'
#     kwargs_light = {'magnitude': 17.5664222662219}
#     amp_light_kwargs = StrongLens._get_amp_light_kwargs(magnitude_zero_point, light_model_class, kwargs_light)
#     assert amp_light_kwargs == {'amp': 0.0017782794100389228}

# def test_build_kwargs_light_dict():
#     mag_dict = {'F106': 17.5664222662219}
#     kwargs_light = {'magnitude': 17.5664222662219}
#     kwargs_light_dict = StrongLens._build_kwargs_light_dict(mag_dict, kwargs_light)
#     assert kwargs_light_dict == {'F106': {'magnitude': 17.5664222662219}}

# def test_validate_mags():
#     lens_mags = {'F106': 17.5664222662219}
#     source_mags = {'F106': 21.434711611915137}
#     StrongLens._validate_mags(lens_mags, source_mags)
#     # Add assertions based on the expected behavior

# def test_convert_magnitudes_to_lenstronomy_amps():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     band = 'F106'
#     lens._convert_magnitudes_to_lenstronomy_amps(band)
#     # Add assertions based on the expected behavior

# def test_set_classes():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._set_classes()
#     # Add assertions based on the expected behavior

# def test_set_lens_cosmo():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._set_lens_cosmo()
#     # Add assertions based on the expected behavior

# def test_set_model():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._set_model()
#     # Add assertions based on the expected behavior

# def test_get_source_pixel_coords():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     coords = (0.0, 0.0)
#     source_pixel_coords = lens.get_source_pixel_coords(coords)
#     assert source_pixel_coords == (0.0, 0.0)  # Replace with expected value

# def test_get_lens_pixel_coords():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     coords = (0.0, 0.0)
#     lens_pixel_coords = lens.get_lens_pixel_coords(coords)
#     assert lens_pixel_coords == (0.0, 0.0)  # Replace with expected value

# def test_mass_physical_to_lensing_units():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._mass_physical_to_lensing_units()
#     # Add assertions based on the expected behavior

# def test_set_up_pixel_grid():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._set_up_pixel_grid()
#     # Add assertions based on the expected behavior

# def test_unpack_kwargs_params():
#     kwargs_params = {'kwargs_lens': [{'center_x': -0.007876281728887604,
#             'center_y': 0.010633393703246008,
#             'e1': 0.004858808997848661,
#             'e2': 0.0075210751726143355,
#             'theta_E': 1.168082477232392},
#             {'dec_0': 0,
#             'gamma1': -0.03648819840013156,
#             'gamma2': -0.06511863424492038,
#             'ra_0': 0},
#             {'dec_0': 0, 'kappa': 0.06020941823541971, 'ra_0': 0}],
# 'kwargs_lens_light': [{'R_sersic': 0.5300707454127908,
#                 'center_x': -0.007876281728887604,
#                 'center_y': 0.010633393703246008,
#                 'e1': 0.023377277902774978,
#                 'e2': 0.05349948216860632,
#                 'magnitude': 17.5664222662219,
#                 'n_sersic': 4.0}],
# 'kwargs_ps': None,
# 'kwargs_source': [{'R_sersic': 0.1651633078964498,
#             'center_x': 0.30298310338567075,
#             'center_y': -0.3505004565139597,
#             'e1': -0.06350855238708408,
#             'e2': -0.08420760408362458,
#             'magnitude': 21.434711611915137,
#             'n_sersic': 1.0}]}
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._unpack_kwargs_params(kwargs_params)
#     # Add assertions based on the expected behavior

# def test_unpack_kwargs_model():
#     kwargs_model = {'cosmo': default_cosmology.get(),
#     'lens_light_model_list': ['SERSIC_ELLIPSE'],
#     'lens_model_list': ['SIE', 'SHEAR', 'CONVERGENCE'],
#     'lens_redshift_list': [0.2902115249535011,
#                         0.2902115249535011,
#                         0.2902115249535011],
#     'source_light_model_list': ['SERSIC_ELLIPSE'],
#     'source_redshift_list': [0.5876899931818929],
#     'z_source': 0.5876899931818929,
#     'z_source_convention': 5}
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     lens._unpack_kwargs_model(kwargs_model)
#     # Add assertions based on the expected behavior

# def test_str():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     str_repr = str(lens)
#     assert str_repr == "StrongLens object"  # Replace with expected value

# def test_csv_row():
#     lens = StrongLens(kwargs_model={}, kwargs_params={}, lens_mags={}, source_mags={}, lens_stellar_mass=None, lens_vel_disp=None, snr=None, uid=None)
#     csv_row = lens.csv_row()
#     assert csv_row == []  # Replace with expected value

# def test_get_csv_headers():
#     headers = StrongLens.get_csv_headers()
#     assert headers == []  # Replace with expected value

def test_einstein_radius_to_velocity_dispersion():
    cosmo = default_cosmology.get()

    einstein_radius = 1.4811764093086500
    z_lens = 0.2478444815301060
    z_source = 1.7249902383698100
    velocity_dispersion = 255.25047210330000
    test_velocity_dispersion = strong_lens.einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source,
                                                                                  cosmo)
    assert velocity_dispersion == approx(test_velocity_dispersion, rel=1e-6)

    einstein_radius = 1.9724261658572900
    z_lens = 0.11804160736612800
    z_source = 1.0785714350247300
    velocity_dispersion = 282.2882717656310
    test_velocity_dispersion = strong_lens.einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source,
                                                                                  cosmo)
    assert velocity_dispersion == approx(test_velocity_dispersion, rel=1e-6)


def test_velocity_dispersion_to_einstein_radius():
    cosmo = default_cosmology.get()

    velocity_dispersion = 255.25047210330000
    z_lens = 0.2478444815301060
    z_source = 1.7249902383698100
    einstein_radius = 1.4811764093086500
    test_einstein_radius = strong_lens.velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source,
                                                                              cosmo)
    assert einstein_radius == approx(test_einstein_radius, rel=1e-6)

    velocity_dispersion = 282.2882717656310
    z_lens = 0.11804160736612800
    z_source = 1.0785714350247300
    einstein_radius = 1.9724261658572900
    test_einstein_radius = strong_lens.velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source,
                                                                              cosmo)
    assert einstein_radius == approx(test_einstein_radius, rel=1e-6)


# TODO def test_mass_to_einstein_radius():

# TODO einstein_radius_to_mass
