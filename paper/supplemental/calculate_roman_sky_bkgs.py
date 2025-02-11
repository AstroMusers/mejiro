import hydra
import json
import os
import speclite.filters
import sys
from lenstronomy.Util import data_util
from slsim.Observations.roman_speclite import configure_roman_filters
from slsim.Observations.roman_speclite import filter_names


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local modules
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    import mejiro
    from mejiro.helpers import convert
    from mejiro.instruments.roman import Roman

    # calculate zeropoints from original Roman Technical Specification filter response curves (not split out by SCA) 
    configure_roman_filters()
    roman_filters = filter_names()
    roman_filters.sort()
    og_roman_filters = [speclite.filters.load_filter(f) for f in roman_filters[:8]]

    original_zeropoints = {}
    for filter in og_roman_filters:
        original_zeropoints[filter.name[-4:]] = convert.get_zeropoint_magnitude(filter.wavelength, filter.response)

    # get zodiacal light and thermal background values from Roman Technical Specification
    roman = Roman()
    min_zodi_counts = roman.min_zodi
    thermal_bkg_counts = roman.thermal_bkg

    # convert sky backgrounds to magnitudes using original Roman Technical Specifications filter response curves
    min_zodi_mags = {key: data_util.cps2magnitude(v, original_zeropoints[key]) for key, v in min_zodi_counts.items()}
    thermal_bkg_mags = {key: data_util.cps2magnitude(v, original_zeropoints[key]) for key, v in
                        thermal_bkg_counts.items()}

    # convert sky backgrounds from magnitudes to counts for each SCA using its zeropoint magnitude
    min_zodi_cps_dict = {}
    thermal_bkg_cps_dict = {}
    for sca, zp_dict in roman.zp_dict.items():
        min_zodi_cps_dict[sca] = {}
        thermal_bkg_cps_dict[sca] = {}
        for filter, zp in zp_dict.items():
            min_zodi_cps_dict[sca][filter] = data_util.magnitude2cps(min_zodi_mags[filter], zp)
            thermal_bkg_cps_dict[sca][filter] = data_util.magnitude2cps(thermal_bkg_mags[filter], zp)

    # save sky background counts to json
    module_path = os.path.dirname(mejiro.__file__)

    min_zodi_output_file = os.path.join(module_path, 'data', 'roman_minimum_zodiacal_light.json')
    json.dump(min_zodi_cps_dict, open(min_zodi_output_file, 'w'), indent=4)
    print(f'Minimum zodiacal light saved to {min_zodi_output_file}')

    thermal_bkg_output_file = os.path.join(module_path, 'data', 'roman_thermal_background.json')
    json.dump(thermal_bkg_cps_dict, open(thermal_bkg_output_file, 'w'), indent=4)
    print(f'Thermal background saved to {thermal_bkg_output_file}')


if __name__ == '__main__':
    main()
