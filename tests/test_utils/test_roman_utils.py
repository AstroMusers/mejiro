import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mejiro.utils import roman_util


@pytest.mark.parametrize('verbose', [False, True])
def test_nancy_bandpass_integration_without_numpy_trapz(monkeypatch, verbose):
    path = Path(__file__).resolve().parents[2] / 'projects' / 'nancy' / 'nancy.py'
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if (
        isinstance(node, ast.ImportFrom) and node.module == 'scipy.integrate'
    ) or (isinstance(node, ast.FunctionDef) and node.name == 'convert_to_cps')]
    wavelengths = np.geomspace(0.4, 2.0, 20)  # [micron]
    brightness = np.linspace(1.0, 3.0, wavelengths.size)  # [MJy/sr]
    response = SimpleNamespace(
        wavelength=np.array([3000.0, 4000.0, 5500.0, 8500.0, 11000.0, 15000.0, 18000.0, 22000.0]),  # [Angstrom]
        response=np.array([0.0, 0.2, 0.4, 0.8, 1.0, 0.7, 0.3, 0.0]),
    )
    roman = SimpleNamespace(load_speclite_filters=lambda: [None] * 4 + [response])
    namespace = {'np': np, 'Roman': roman}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    wavelength_angstrom = wavelengths * 1e4  # [Angstrom]
    flux_nu = brightness * (0.11 / 206265.0)**2 * 1e-17  # [erg/s/cm^2/Hz/pixel]
    flux_lambda = flux_nu * 2.99792458e18 / wavelength_angstrom**2  # [erg/s/cm^2/Angstrom/pixel]
    photon_flux = flux_lambda * wavelength_angstrom / (6.62607015e-27 * 2.99792458e18)  # [photon/s/cm^2/Angstrom/pixel]
    integrand = np.interp(response.wavelength, wavelength_angstrom, photon_flux, left=0, right=0) * response.response
    expected = np.sum(np.diff(response.wavelength) * (integrand[1:] + integrand[:-1]) / 2.0) * 3.60767e4
    monkeypatch.delattr(np, 'trapz', raising=False)

    actual = namespace['convert_to_cps'](wavelengths, brightness, verbose=verbose)

    assert actual == pytest.approx(expected, rel=1e-14)


def test_divide_up_sca():
    # sides = 56
    centers = roman_util.divide_up_sca(56)
    result = [centers[0], centers[1], centers[2], centers[-1]]
    expected = [(36, 36), (36, 109), (36, 182), (4051, 4051)]
    assert result == expected, f"Expected {expected}, but got {result}"

    # sides = 3 (4088 cannot be evenly divided by 3)
    with pytest.raises(AssertionError, match='Sub-array size must be a whole number'):
        roman_util.divide_up_sca(3)

    # unhappy path
    with pytest.raises(AssertionError):
        roman_util.divide_up_sca(0)


def test_get_sca_string():
    # test ints
    assert roman_util.get_sca_string(1) == 'SCA01'
    assert roman_util.get_sca_string(10) == 'SCA10'

    # test floats
    assert roman_util.get_sca_string(1.0) == 'SCA01'
    assert roman_util.get_sca_string(10.0) == 'SCA10'

    # test strings
    assert roman_util.get_sca_string('1') == 'SCA01'
    assert roman_util.get_sca_string('10') == 'SCA10'
    assert roman_util.get_sca_string('01') == 'SCA01'
    assert roman_util.get_sca_string('SCA01') == 'SCA01'
    assert roman_util.get_sca_string('SCA10') == 'SCA10'
    assert roman_util.get_sca_string('SCA1') == 'SCA01'


def test_get_sca_int():
    # test ints
    assert roman_util.get_sca_int(1) == 1
    assert roman_util.get_sca_int(10) == 10

    # test floats
    assert roman_util.get_sca_int(1.0) == 1
    assert roman_util.get_sca_int(10.0) == 10

    # test strings
    assert roman_util.get_sca_int('1') == 1
    assert roman_util.get_sca_int('10') == 10
    assert roman_util.get_sca_int('01') == 1
    assert roman_util.get_sca_int('SCA01') == 1
    assert roman_util.get_sca_int('SCA10') == 10
    assert roman_util.get_sca_int('SCA1') == 1


def test_translate_band():
    # test valid bands
    assert roman_util.translate_band('F062') == 'F062'
    assert roman_util.translate_band('F087') == 'F087'
    assert roman_util.translate_band('F106') == 'F106'
    assert roman_util.translate_band('F129') == 'F129'
    assert roman_util.translate_band('F158') == 'F158'
    assert roman_util.translate_band('F184') == 'F184'
    assert roman_util.translate_band('F146') == 'F146'
    assert roman_util.translate_band('F213') == 'F213'

    # test valid aliases
    assert roman_util.translate_band('R') == 'F062'
    assert roman_util.translate_band('Z') == 'F087'
    assert roman_util.translate_band('Y') == 'F106'
    assert roman_util.translate_band('J') == 'F129'
    assert roman_util.translate_band('H') == 'F158'
    assert roman_util.translate_band('H/K') == 'F184'
    assert roman_util.translate_band('WIDE') == 'F146'
    assert roman_util.translate_band('KS') == 'F213'

    # test invalid band
    try:
        roman_util.translate_band('X')
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # test invalid alias
    try:
        roman_util.translate_band('A')
        assert False, "Expected ValueError"
    except ValueError:
        pass
