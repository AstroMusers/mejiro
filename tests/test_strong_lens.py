import numpy as np
import pytest

from astropy.cosmology import default_cosmology
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from pyHalo.preset_models import preset_model_from_name

from mejiro.galaxy_galaxy import GalaxyGalaxy, SampleGG, SampleSL2S, SampleBELLS


def _make_lens(physical_params=None, use_jax=None, omit_cosmo=False,
               kwargs_lens_light_extra=None, kwargs_source_extra=None):
    """Build a minimal valid GalaxyGalaxy for tests that need controlled inputs.

    By default the magnitudes dict covers both lens and source so
    `validate_light_models` skips the 'amp' requirement on the light kwargs.
    Pass `physical_params` to replace it entirely; pass `omit_cosmo=True` to
    leave 'cosmo' out of kwargs_model (for the constructor's validation path).
    """
    if physical_params is None:
        physical_params = {
            'magnitudes': {
                'lens': {'F062': 17.0},
                'source': {'F062': 21.0},
            }
        }
    kwargs_model = {
        'lens_light_model_list': ['SERSIC_ELLIPSE'],
        'lens_model_list': ['SIE'],
        'lens_redshift_list': [0.3],
        'source_light_model_list': ['SERSIC_ELLIPSE'],
        'source_redshift_list': [1.5],
        'z_source': 1.5,
    }
    if not omit_cosmo:
        kwargs_model['cosmo'] = default_cosmology.get()
    base_lens_light = {'R_sersic': 0.5, 'n_sersic': 4.0,
                       'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}
    base_source = {'R_sersic': 0.2, 'n_sersic': 1.0,
                   'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}
    if kwargs_lens_light_extra:
        base_lens_light.update(kwargs_lens_light_extra)
    if kwargs_source_extra:
        base_source.update(kwargs_source_extra)
    kwargs_params = {
        'kwargs_lens': [{'theta_E': 1.0, 'e1': 0.0, 'e2': 0.0,
                         'center_x': 0.0, 'center_y': 0.0}],
        'kwargs_lens_light': [base_lens_light],
        'kwargs_source': [base_source],
    }
    return GalaxyGalaxy(
        name='test',
        coords=None,
        kwargs_model=kwargs_model,
        kwargs_params=kwargs_params,
        physical_params=physical_params,
        use_jax=use_jax,
    )


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


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

def test_init_requires_cosmo_in_kwargs_model():
    with pytest.raises(ValueError, match="cosmo"):
        _make_lens(omit_cosmo=True)


def test_init_use_jax_defaults_to_all_false():
    sl = _make_lens(use_jax=None)
    assert sl.use_jax == [False] * len(sl.lens_model_list)


def test_init_use_jax_bool_expands_to_list():
    true_lens = _make_lens(use_jax=True)
    false_lens = _make_lens(use_jax=False)
    assert true_lens.use_jax == [True] * len(true_lens.lens_model_list)
    assert false_lens.use_jax == [False] * len(false_lens.lens_model_list)


def test_init_use_jax_list_preserved():
    sl = _make_lens(use_jax=[True])   # 1 lens model in _make_lens
    assert sl.use_jax == [True]


def test_init_use_jax_list_length_mismatch_raises():
    # _make_lens has 1 lens_model_list entry; passing a 2-element list mismatches
    with pytest.raises(ValueError, match="must match the number of lens models"):
        _make_lens(use_jax=[True, False])


def test_init_use_jax_invalid_type_raises():
    with pytest.raises(ValueError, match="boolean or a list of booleans"):
        _make_lens(use_jax="yes")


# ---------------------------------------------------------------------------
# magnitude / maggies API
# ---------------------------------------------------------------------------

def test_get_magnitude_returns_value_for_each_kind(sample_gg):
    assert sample_gg.get_lens_magnitude('F062') == 17.9
    assert sample_gg.get_source_magnitude('F062') == 21.9
    assert sample_gg.get_magnitude('lens', 'F129') == 17.3
    assert sample_gg.get_magnitude('source', 'F184') == 20.5


def test_get_magnitude_raises_when_magnitudes_missing(sample_gg):
    sample_gg.physical_params = {}   # wipe magnitudes after init
    with pytest.raises(ValueError, match="Magnitudes are not provided"):
        sample_gg.get_magnitude('lens', 'F062')


def test_get_magnitude_raises_when_kind_missing(sample_gg):
    # SampleGG has 'lens' and 'source' but not 'lensed_source'
    with pytest.raises(ValueError, match="lensed_source magnitudes are not provided"):
        sample_gg.get_lensed_source_magnitude('F062')


def test_get_magnitude_raises_when_band_missing(sample_gg):
    with pytest.raises(ValueError, match="band NOT_A_BAND"):
        sample_gg.get_lens_magnitude('NOT_A_BAND')


def test_get_maggies_is_pogson_formula(sample_gg):
    # maggies = 10 ** (-0.4 * mag); SampleGG lens F062 = 17.9
    assert sample_gg.get_maggies('lens', 'F062') == pytest.approx(10 ** (-0.4 * 17.9))
    assert sample_gg.get_maggies('source', 'F184') == pytest.approx(10 ** (-0.4 * 20.5))


# ---------------------------------------------------------------------------
# physical-parameter retrieval and fallbacks
# ---------------------------------------------------------------------------

def test_get_velocity_dispersion_raises_when_missing(sample_gg):
    # SampleGG does not store lens_velocity_dispersion
    with pytest.raises(ValueError, match="Velocity dispersion not found"):
        sample_gg.get_velocity_dispersion()


def test_get_velocity_dispersion_returns_value(sample_gg):
    sample_gg.physical_params['lens_velocity_dispersion'] = 250.0
    assert sample_gg.get_velocity_dispersion() == 250.0


def test_get_stellar_mass_raises_when_missing(sample_gg):
    with pytest.raises(ValueError, match="Stellar mass not found"):
        sample_gg.get_stellar_mass()


def test_get_stellar_mass_returns_value(sample_gg):
    sample_gg.physical_params['lens_stellar_mass'] = 1.5e11
    assert sample_gg.get_stellar_mass() == 1.5e11


def test_get_main_halo_mass_returns_stored_value(sample_gg):
    # SampleGG sets main_halo_mass = 10**13.4 directly
    assert sample_gg.get_main_halo_mass() == pytest.approx(10 ** 13.4)


def test_get_main_halo_mass_falls_back_to_stellar_mass():
    sl = _make_lens(physical_params={
        'magnitudes': {'lens': {'F062': 17.0}, 'source': {'F062': 21.0}},
        'lens_stellar_mass': 1e11,
    })
    np.random.seed(0)   # stellar_to_main_halo_mass(sample=True) draws from truncnorm
    halo_mass = sl.get_main_halo_mass()
    # bounds from cosmo.stellar_to_main_halo_mass: 1e11 * alpha * (1+z)**beta
    # with alpha in [15, 87] and beta in [-0.9, 2.7] at z=0.3
    assert np.isfinite(halo_mass)
    assert halo_mass > 0
    assert 1e11 * 15 * (1.3 ** -0.9) <= halo_mass <= 1e11 * 87 * (1.3 ** 2.7)


def test_get_main_halo_mass_raises_when_no_basis():
    sl = _make_lens(physical_params={
        'magnitudes': {'lens': {'F062': 17.0}, 'source': {'F062': 21.0}},
    })
    with pytest.raises(ValueError, match="main_halo_mass.*lens_stellar_mass"):
        sl.get_main_halo_mass()


# ---------------------------------------------------------------------------
# Einstein radius retrieval
# ---------------------------------------------------------------------------

def test_get_einstein_radius_from_kwargs_lens(sample_gg):
    # SampleGG has theta_E = 1.168... in kwargs_lens[0] and no einstein_radius
    assert sample_gg.get_einstein_radius() == pytest.approx(1.168082477232392)


def test_get_einstein_radius_prefers_physical_params(sample_gg):
    sample_gg.physical_params['einstein_radius'] = 1.42
    assert sample_gg.get_einstein_radius() == 1.42


def test_get_einstein_radius_raises_when_unavailable(sample_gg):
    sample_gg.physical_params.pop('einstein_radius', None)
    sample_gg.kwargs_lens[0].pop('theta_E')   # SIE without theta_E
    with pytest.raises(ValueError, match="einstein_radius.*theta_E"):
        sample_gg.get_einstein_radius()


# ---------------------------------------------------------------------------
# get_lens_cosmo: lazy initialization + caching
# ---------------------------------------------------------------------------

def test_get_lens_cosmo_lazy_and_cached(sample_gg):
    assert sample_gg.lens_cosmo is None   # not built in __init__
    first = sample_gg.get_lens_cosmo()
    assert isinstance(first, LensCosmo)
    second = sample_gg.get_lens_cosmo()
    assert second is first         # cached, not rebuilt


# ---------------------------------------------------------------------------
# kwargs_* / lens_* property getters and setters
# ---------------------------------------------------------------------------

def test_kwargs_lens_getter_and_setter_round_trip(sample_gg):
    original = sample_gg.kwargs_lens
    assert original is sample_gg.kwargs_params['kwargs_lens']

    new_value = [{'theta_E': 2.0, 'e1': 0.0, 'e2': 0.0,
                  'center_x': 0.0, 'center_y': 0.0}]
    sample_gg.kwargs_lens = new_value
    assert sample_gg.kwargs_lens is new_value
    assert sample_gg.kwargs_params['kwargs_lens'] is new_value


def test_lens_model_list_setter_writes_through_to_kwargs_model(sample_gg):
    sample_gg.lens_model_list = ['SIS']
    assert sample_gg.kwargs_model['lens_model_list'] == ['SIS']


def test_lens_model_property_returns_lenstronomy_objects(sample_gg):
    assert isinstance(sample_gg.lens_model, LensModel)
    assert isinstance(sample_gg.lens_light_model, LightModel)
    assert isinstance(sample_gg.source_light_model, LightModel)


# ---------------------------------------------------------------------------
# validate_light_models branches
# ---------------------------------------------------------------------------

def test_validate_light_models_returns_false_when_magnitudes_complete(sample_gg):
    # SampleGG provides magnitudes for both lens and source
    assert sample_gg.validate_light_models() is False


def test_validate_light_models_returns_true_when_amps_provided():
    sl = _make_lens(
        physical_params={},   # no magnitudes -> 'amp' is required
        kwargs_lens_light_extra={'amp': 1.0},
        kwargs_source_extra={'amp': 1.0},
    )
    assert sl.validate_light_models() is True


def test_validate_light_models_raises_when_amp_missing_and_no_magnitudes():
    with pytest.raises(ValueError, match="Missing 'amp'"):
        _make_lens(physical_params={})   # neither magnitudes nor 'amp'


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

def test_str_includes_identifying_fields(sample_gg):
    s = str(sample_gg)
    assert 'SampleGG' in s
    assert str(sample_gg.z_lens) in s
    assert str(sample_gg.z_source) in s
