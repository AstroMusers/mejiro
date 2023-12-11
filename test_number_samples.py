import os
import sys
import numpy as np
from tqdm import tqdm

from package.helpers import test_physical_lens, pyhalo
from package.pandeia import pandeia_input


def main():
    cwd = os.getcwd()
    while os.path.basename(os.path.normpath(cwd)) != 'roman-pandeia':
        cwd = os.path.dirname(cwd)
    repo_path = cwd
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    csv = os.path.join(repo_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    array_dir = os.path.join(repo_path, 'output', 'arrays', 'diagnostics', 'num_samples')

    num_samples_list = [10e2, 5*10e2, 10e3, 5*10e3, 10e4, 5*10e4, 10e5]
    num_samples_list = [int(i) for i in num_samples_list]  # convert to list of int as scientific notation in Python gives float
    grid_oversample = 9
    execution_times, point_source_count, estimated_times = [], [], []

    # use test lens
    lens = test_physical_lens.TestPhysicalLens()

    # add CDM subhalos
    lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))

    for num_samples in tqdm(num_samples_list):
        # generate lenstronomy model, varying grid oversample factor
        model = lens.get_array(num_pix=45 * grid_oversample, side=5.)

        # build Pandeia input
        calc, num_point_sources = pandeia_input.build_pandeia_calc(csv=csv,
                                                array=model, 
                                                lens=lens, 
                                                band='f106', 
                                                num_samples=num_samples)
        
        # get estimated calculation time
        estimated_times.append(pandeia_input.estimate_calculation_time(num_point_sources))

        # do Pandeia calculation        
        image, execution_time = pandeia_input.get_pandeia_image(calc)
        execution_times.append(execution_time)
        point_source_count.append(num_point_sources)
        
        # save detector image
        np.save(os.path.join(array_dir, f'num_samples_{num_samples}'), image)

    # save execution times and point source counts
    np.save(os.path.join(array_dir, 'execution_times_num_samples.npy'), execution_times)
    np.save(os.path.join(array_dir, 'point_source_count_num_samples.npy'), point_source_count)
    np.save(os.path.join(array_dir, 'estimated_time_num_samples.npy'), estimated_times)


if __name__ == '__main__':
    main()
