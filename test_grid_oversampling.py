import os
import sys
import numpy as np
from tqdm import tqdm

from package.helpers import test_physical_lens
from package.pandeia import pandeia_input
from package.pandeia.pandeia_output import PandeiaOutput
from package.helpers import pyhalo


def main():
    cwd = os.getcwd()
    while os.path.basename(os.path.normpath(cwd)) != 'roman-pandeia':
        cwd = os.path.dirname(cwd)
    repo_path = cwd
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    csv = os.path.join(repo_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    array_dir = os.path.join(repo_path, 'output', 'arrays', 'diagnostics', 'grid_oversampling')

    num_samples = 100000
    grid_oversample_list = [1, 3, 5, 7, 9, 11]
    execution_times, point_source_count = [], []

    for grid_oversample in tqdm(grid_oversample_list):
        # use test lens
        lens = test_physical_lens.TestPhysicalLens()

        # add CDM subhalos
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))

        # generate lenstronomy model, varying grid oversample factor
        model = lens.get_array(num_pix=45 * grid_oversample, side=5.)

        # build Pandeia input
        calc, num_point_sources = pandeia_input.build_pandeia_calc(csv=csv,
                                                array=model, 
                                                lens=lens, 
                                                band='f106', 
                                                num_samples=num_samples)

        # do Pandeia calculation        
        image, execution_time = pandeia_input.get_pandeia_image(calc)
        execution_times.append(execution_time)
        point_source_count.append(num_point_sources)
        
        # save detector image
        np.save(os.path.join(array_dir, f'grid_oversampling_{grid_oversample}'), image)

    # save execution times
    np.save(os.path.join(array_dir, 'execution_times_grid_oversampling.npy'), execution_times)
    np.save(os.path.join(array_dir, 'point_source_count_grid_oversampling.npy'), point_source_count)


if __name__ == '__main__':
    main()
