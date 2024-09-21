from mejiro.instruments.roman import Roman


def test_init():
    roman = Roman()

    assert roman.name == 'Roman'
    assert roman.pixels_per_axis == 4088

    # check that all files loaded
    assert not roman.df.empty, 'roman_spacecraft_and_instrument_parameters DataFrame is empty'
    assert roman.zp_dict, 'zp_dict is empty'  # NB empty dictionaries evaluate to False
    assert roman.min_zodi_dict, 'min_zodi_dict is empty'
    assert roman.thermal_bkg_dict, 'thermal_bkg_dict is empty'
    