import os
import sys

import hydra
import numpy as np
from tqdm import tqdm

from package.helpers import pyhalo, pandeia_input
from package.lenses import test_physical_lens
from package.utils import util


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir
    array_dir = os.path.join(array_dir, 'sample_skypy_lens')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get Roman pixel scale
    # csv = os.path.join(repo_dir, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    # roman_pixel_scale = roman_params.RomanParameters(csv).get_pixel_scale()

    # num_pix = 51  # (45 + (2 * 3))
    # side = 5.61  # (4.95 + (2 * 0.33))
    # grid_oversample = 5
    num_samples_list = [10, 100, 1000, 10000, 100000, 1000000]
    num_samples_list = [int(i) for i in
                        num_samples_list]  # convert to list of int as scientific notation in Python gives float

    # use test lens
    lens = test_physical_lens.TestPhysicalLens()
    lens_list = [lens] * len(num_samples_list)

    # add CDM subhalos; NB same subhalo population for all
    lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))

    # # split up the lenses into batches based on core count
    # cpu_count = multiprocessing.cpu_count()
    # process_count = len(num_samples_list)

    # # process the batches
    # pool = Pool(processes=process_count) 
    # for i, output in enumerate(pool.starmap(generate, zip(lens_list, num_samples_list))):
    #     (image) = output
    #     np.save(os.path.join(array_dir, f'sample_skypy_lens_{num_samples_list[i]}'), image)

    for num_samples in tqdm(num_samples_list):
        image = generate(lens, num_samples)
        np.save(os.path.join(array_dir, f'sample_skypy_lens_{num_samples}'), image)


def generate(lens, num_samples):
    # (lens, num_samples) = input
    model = lens.get_array(num_pix=51 * 5, side=5.61)  # .get_array(num_pix=97, side=10.67)

    # build Pandeia input
    calc, _ = pandeia_input.build_pandeia_calc(
        csv='/nfshome/bwedig/roman-pandeia/data/roman_spacecraft_and_instrument_parameters.csv',
        array=model,
        lens=lens,
        band='f106',
        num_samples=num_samples,
        suppress_output=True)

    # do Pandeia calculation        
    image, _ = pandeia_input.get_pandeia_image(calc, suppress_output=True)

    return image


if __name__ == '__main__':
    main()
