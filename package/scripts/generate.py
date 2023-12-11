import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from package.helpers.lens import Lens
from package.helpers import pyhalo
from package.pandeia import pandeia_input


def main(lens):   
    csv = os.path.join('nfshome', 'bwedig', 'roman-pandeia', 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    roman_params = pd.read_csv(csv)
    roman_pixel_scale = float(roman_params.loc[roman_params['Name'] == 'WFI_Pixel_Scale']['Value'].to_string(index=False))

    # add CDM subhalos
    try:
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))
    except:
        return 'potato'

    grid_oversample = 1
    num_samples = 100

    buffer = 0.5
    side = 10.
    num_pix = round(side / roman_pixel_scale) * grid_oversample
    # if num_pix even, need to make it odd
    if num_pix % 2 == 0:
        num_pix += 1

    # build model
    model = lens.get_array(num_pix=num_pix, side=side + buffer)

    # build Pandeia input
    calc, num_point_sources = pandeia_input.build_pandeia_calc(csv=csv,
                                            array=model, 
                                            lens=lens,
                                            side=side, 
                                            band='f106', 
                                            num_samples=num_samples)

    # do Pandeia calculation        
    image, execution_time = pandeia_input.get_pandeia_image(calc)

    return image, execution_time, num_point_sources


if __name__ == '__main__':
    main()
