import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock
from astropy.cosmology import default_cosmology

from mejiro.lensed_supernova import LensedSupernova
from mejiro.galaxy_galaxy import GalaxyGalaxy
from mejiro.strong_lens import StrongLens


# --- Helpers ---

def _make_kwargs_model(z_lens=0.5, z_source=1.5):
    return {
        'cosmo': default_cosmology.get(),
        'lens_light_model_list': ['SERSIC_ELLIPSE'],
        'lens_model_list': ['SIE', 'SHEAR', 'CONVERGENCE'],
        'source_light_model_list': ['SERSIC_ELLIPSE'],
        'point_source_model_list': ['LENSED_POSITION'],
        'lens_redshift_list': [z_lens, z_lens, z_lens],
        'source_redshift_list': [z_source],
        'z_source': z_source,
    }


def _make_kwargs_params(z_lens=0.5, z_source=1.5):
    return {
        'kwargs_lens': [
            {'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0},
            {'gamma1': 0.01, 'gamma2': -0.01, 'ra_0': 0, 'dec_0': 0},
            {'kappa': 0.01, 'ra_0': 0, 'dec_0': 0},
        ],
        'kwargs_lens_light': [
            {'magnitude': 20.0, 'R_sersic': 1.0, 'n_sersic': 4.0,
             'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0},
        ],
        'kwargs_source': [
            {'magnitude': 23.0, 'R_sersic': 0.3, 'n_sersic': 1.0,
             'center_x': 0.1, 'center_y': -0.1, 'e1': 0.0, 'e2': 0.0},
        ],
        'kwargs_ps': [
            {
                'ra_image': np.array([0.5, -0.5, 0.3, -0.3]),
                'dec_image': np.array([0.3, -0.3, -0.5, 0.5]),
                'magnitude': [25.0, 25.5, 26.0, 26.5],
            },
        ],
    }


def _make_physical_params():
    time_arr = np.linspace(-20, 80, 50)
    # simple light curves: 4 images, each offset by 1 mag
    mags_per_image = [
        25.0 - 2.5 * np.exp(-(time_arr - 10)**2 / 200),
        25.5 - 2.5 * np.exp(-(time_arr - 10)**2 / 200),
        26.0 - 2.5 * np.exp(-(time_arr - 10)**2 / 200),
        26.5 - 2.5 * np.exp(-(time_arr - 10)**2 / 200),
    ]
    return {
        'einstein_radius': 1.0,
        'lens_stellar_mass': 1e11,
        'lens_velocity_dispersion': 250.0,
        'magnification': 5.0,
        'magnitudes': {
            'lens': {'F129': 20.0},
            'source': {'F129': 23.0},
            'lensed_source': {'F129': 21.5},
        },
        'sn_type': 'Ia',
        'time_delays': np.array([0.0, 5.2, 12.3, 15.1]),
        'image_magnifications': np.array([3.5, -2.1, 1.8, -0.5]),
        'light_curves': {
            'F129': {
                'time': time_arr,
                'magnitudes': mags_per_image,
            },
        },
    }


def _make_lensed_supernova():
    return LensedSupernova(
        name='test_sn',
        coords=None,
        kwargs_model=_make_kwargs_model(),
        kwargs_params=_make_kwargs_params(),
        physical_params=_make_physical_params(),
    )


def _make_mock_slsim_lens(bands=None):
    """Create a MagicMock that mimics an SLSim Lens with PointPlusExtendedSource."""
    if bands is None:
        bands = ['F129']

    mock = MagicMock()
    mock.cosmo = default_cosmology.get()
    mock.deflector_redshift = 0.5
    mock.source_redshift_list = [1.5]
    mock.deflector._deflector._deflector_dict = {f'mag_{b}': 20.0 for b in bands}

    lightcurve_time = np.linspace(-20, 80, 50)
    n_images = 4

    def fake_lenstronomy_kwargs(band, time=None):
        return _make_kwargs_model(), _make_kwargs_params()

    mock.lenstronomy_kwargs.side_effect = fake_lenstronomy_kwargs
    mock.deflector_magnitude.return_value = 20.0
    mock.extended_source_magnitude.return_value = [23.0]
    mock.extended_source_magnification = [5.0]
    mock.einstein_radius = [1.0]
    mock.deflector_stellar_mass.return_value = 1e11
    mock.deflector_velocity_dispersion.return_value = 250.0
    mock.deflector.deflector_type = 'EPL'

    # Point source methods
    base_curve = 25.0 - 2.5 * np.exp(-(lightcurve_time - 10)**2 / 200)
    mock.point_source_magnitude.return_value = [
        [base_curve + i * 0.5 for i in range(n_images)]
    ]
    mock.point_source_arrival_times.return_value = [np.array([0.0, 5.2, 12.3, 15.1])]
    mock.point_source_magnification.return_value = [np.array([3.5, -2.1, 1.8, -0.5])]

    # SN metadata via source_dict on extended source
    mock._source.__getitem__.return_value._source._extended_source.source_dict = {
        'sn_type': 'Ia',
        'lightcurve_time': lightcurve_time,
        'sn_absolute_mag_band': 'bessellb',
        'sn_absolute_zpsys': 'ab',
        'kwargs_variability': {'supernovae_lightcurve'} | set(bands),
    }

    return mock


# --- Tests ---

def test_init():
    sn = _make_lensed_supernova()

    assert sn.name == 'test_sn'
    assert sn.sn_type == 'Ia'
    assert sn.z_source == 1.5
    assert sn.z_lens == 0.5
    assert isinstance(sn, LensedSupernova)
    assert isinstance(sn, GalaxyGalaxy)
    assert isinstance(sn, StrongLens)


def test_get_time_delays():
    sn = _make_lensed_supernova()
    delays = sn.get_time_delays()
    np.testing.assert_array_almost_equal(delays, [0.0, 5.2, 12.3, 15.1])


def test_get_time_delays_missing():
    sn = _make_lensed_supernova()
    del sn.physical_params['time_delays']
    with pytest.raises(ValueError, match="Time delays not found"):
        sn.get_time_delays()


def test_get_point_source_magnification():
    sn = _make_lensed_supernova()
    mags = sn.get_point_source_magnification()
    np.testing.assert_array_almost_equal(mags, [3.5, -2.1, 1.8, -0.5])


def test_get_point_source_magnification_missing():
    sn = _make_lensed_supernova()
    del sn.physical_params['image_magnifications']
    with pytest.raises(ValueError, match="Image magnifications not found"):
        sn.get_point_source_magnification()


def test_get_sn_image_positions():
    sn = _make_lensed_supernova()
    ra, dec = sn.get_sn_image_positions()
    np.testing.assert_array_almost_equal(ra, [0.5, -0.5, 0.3, -0.3])
    np.testing.assert_array_almost_equal(dec, [0.3, -0.3, -0.5, 0.5])


def test_get_sn_image_positions_missing():
    sn = _make_lensed_supernova()
    sn.kwargs_params['kwargs_ps'] = []
    with pytest.raises(ValueError, match="No point source parameters found"):
        sn.get_sn_image_positions()


def test_get_light_curve():
    sn = _make_lensed_supernova()
    lc = sn.get_light_curve('F129')
    assert 'time' in lc
    assert 'magnitudes' in lc
    assert len(lc['magnitudes']) == 4  # 4 images
    assert len(lc['time']) == 50


def test_get_light_curve_missing_band():
    sn = _make_lensed_supernova()
    with pytest.raises(ValueError, match="No light curve found for band 'F062'"):
        sn.get_light_curve('F062')


def test_set_observation_time():
    sn = _make_lensed_supernova()

    # set observation time to t=10 (peak of the Gaussian)
    sn.set_observation_time(10.0, 'F129')

    updated_mags = sn.kwargs_ps[0]['magnitude']
    assert len(updated_mags) == 4

    # at t=10, each image should be at its brightest (lowest magnitude)
    # image 0: 25.0 - 2.5 = 22.5, image 1: 25.5 - 2.5 = 23.0, etc.
    expected = [22.5, 23.0, 23.5, 24.0]
    for actual, exp in zip(updated_mags, expected):
        assert actual == pytest.approx(exp, abs=0.1)


def test_set_observation_time_updates_kwargs():
    sn = _make_lensed_supernova()

    # initial magnitudes
    initial_mags = sn.kwargs_ps[0]['magnitude'].copy()

    # set to a different time (far from peak)
    sn.set_observation_time(-20.0, 'F129')
    new_mags = sn.kwargs_ps[0]['magnitude']

    # magnitudes should be different from initial
    assert new_mags != initial_mags


def test_from_slsim():
    mock = _make_mock_slsim_lens(bands=['F129'])
    sn = LensedSupernova.from_slsim(mock, name='test_slsim', bands=['F129'])

    assert isinstance(sn, LensedSupernova)
    assert sn.name == 'test_slsim'
    assert sn.z_lens == 0.5
    assert sn.z_source == 1.5
    assert sn.sn_type == 'Ia'

    # check physical params
    assert 'time_delays' in sn.physical_params
    assert 'image_magnifications' in sn.physical_params
    assert 'light_curves' in sn.physical_params
    assert 'F129' in sn.light_curves

    # check kwargs_ps populated
    assert len(sn.kwargs_ps) > 0
    assert 'point_source_model_list' in sn.kwargs_model
    assert sn.kwargs_model['point_source_model_list'] == ['LENSED_POSITION']


def test_from_slsim_magnitudes():
    mock = _make_mock_slsim_lens(bands=['F129'])
    sn = LensedSupernova.from_slsim(mock, name='test_mags', bands=['F129'])

    mags = sn.physical_params['magnitudes']
    assert mags['lens']['F129'] == 20.0
    assert mags['source']['F129'] == 23.0


def test_from_slsim_time_delays():
    mock = _make_mock_slsim_lens(bands=['F129'])
    sn = LensedSupernova.from_slsim(mock, name='test_td', bands=['F129'])

    np.testing.assert_array_almost_equal(
        sn.get_time_delays(), [0.0, 5.2, 12.3, 15.1]
    )


def test_inherits_galaxy_galaxy_methods():
    sn = _make_lensed_supernova()

    # get_magnification from GalaxyGalaxy
    assert sn.get_magnification() == 5.0

    # get_einstein_radius from StrongLens
    assert sn.get_einstein_radius() == 1.0

    # get_velocity_dispersion from StrongLens
    assert sn.get_velocity_dispersion() == 250.0

    # get_stellar_mass from StrongLens
    assert sn.get_stellar_mass() == 1e11
