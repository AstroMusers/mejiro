import os
import sys

import hydra
import json
import numpy as np
import speclite.filters
from glob import glob
from tqdm import tqdm


def get_zeropoint_magnitude(wavelength, response, effective_area=4.5 * 1e4):
    '''
    see Section 6.1 of [this paper](https://www.aanda.org/articles/aa/full_html/2022/06/aa42897-21/aa42897-21.html) by the Euclid collaboration for explanation of this function

    Roman's collecting area (4.5 m^2) retrieved 16 August 2024 from https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/introduction-to-wfi/wfi-quick-reference
    '''
    # effective area in cm^2

    # assert that wavelength values are evenly spaced
    assert np.allclose(np.diff(wavelength), np.diff(wavelength)[0])

    dv = np.diff(wavelength)[0]
    integral = 0
    for wl, resp in zip(wavelength, response):
        integral += (dv * (1 / wl) * resp)
    
    return 8.9 + (2.5 * np.log10(((effective_area * 1e-23) / (6.602 * 1e-27)) * integral))


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local modules
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    import mejiro

    # set path to filter response curves
    module_path = os.path.dirname(mejiro.__file__)
    filter_responses_dir = os.path.join(module_path, 'data', 'filter_responses')
    print(f'Retrieving filter response curves from {filter_responses_dir}')

    zp_dict = {}
    for sca in tqdm(range(1, 19)):
        group_name = f'RomanSCA{str(sca).zfill(2)}'

        # NB I'm not loading any of the grism or prism stuff here. I can exclude them by sorting the globbed files and selecting the first eight, which are the WFI bands
        filter_response_files = sorted(glob(f'{filter_responses_dir}/{group_name}*.ecsv'))
        roman_filters = [speclite.filters.load_filter(f) for f in filter_response_files[:8]]
        roman_bands = [f.name.split('-')[1] for f in roman_filters]

        sca_dict = {}
        for band, filter in zip(roman_bands, roman_filters):
            sca_dict[band] = get_zeropoint_magnitude(filter.wavelength, filter.response)
        
        zp_dict[f'SCA{str(sca).zfill(2)}'] = sca_dict
    
    output_file = os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')
    json.dump(zp_dict, open(output_file, 'w'), indent=4)
    print(f'Zeropoint magnitudes saved to {output_file}')


if __name__ == '__main__':
    main()
