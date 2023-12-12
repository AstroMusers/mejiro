import os

from package.helpers import pyhalo, roman_params, pandeia_input


def main(lens):
    csv = os.path.join('/nfshome', 'bwedig', 'roman-pandeia', 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    roman_pixel_scale = roman_params.RomanParameters(csv).get_pixel_scale()

    # add CDM subhalos
    try:
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))
    except:
        # traceback.print_exc()
        return None, None, None

    grid_oversample = 3
    num_samples = 100000

    buffer = 0.5
    side = 10.
    num_pix = round((side + buffer) / roman_pixel_scale) * grid_oversample
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
                                                               num_samples=num_samples,
                                                               suppress_output=True)

    # do Pandeia calculation        
    image, execution_time = pandeia_input.get_pandeia_image(calc, suppress_output=True)

    return image, execution_time, num_point_sources


if __name__ == '__main__':
    main()
