import hydra
import numpy as np
import os
import sys
from tqdm import tqdm

from mejiro.helpers import pandeia_input
from mejiro.lenses.test import SampleStrongLens
from mejiro.utils import util


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(config):
    array_dir, pickle_dir, repo_dir = config.machine.array_dir, config.machine.pickle_dir, config.machine.repo_dir

    util.create_directory_if_not_exists(os.path.join(array_dir, 'sample_skypy_lens'))
    util.create_directory_if_not_exists(os.path.join(pickle_dir, 'pyhalo'))

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # use test lens
    lens = SampleStrongLens()
    # lens_list = [lens] * len(num_samples_list)

    # add CDM subhalos; NB same subhalo population for all
    # with open(os.path.join(pickle_dir, 'cdm_subhalos_for_sample_skypy_lens'), 'rb') as results_file:
    #     realizationCDM = pickle.load(results_file)
    # lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(realizationCDM))

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
        np.save(os.path.join(array_dir, f'sample_skypy_lens_no_substructure_{num_samples}'), image)


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
