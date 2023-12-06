import os
import sys
import numpy as np
import pickle

from package.helpers import test_physical_lens
from package.pandeia import pandeia_input
from package.pandeia.pandeia_output import PandeiaOutput


def main():
    cwd = os.getcwd()
    while os.path.basename(os.path.normpath(cwd)) != 'roman-pandeia':
        cwd = os.path.dirname(cwd)
    repo_path = cwd
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    csv = os.path.join(repo_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    array_dir = os.path.join(repo_path, 'output', 'arrays', 'test_physical_lens')
    pickle_dir = os.path.join(repo_path, 'output', 'pickles', 'test_physical_lens')

    oversample_factor_list = [1, 3, 5, 7, 9, 11]
    execution_times = []

    for oversample_factor in oversample_factor_list:
        lens = test_physical_lens.TestPhysicalLens()
        model = lens.get_array(num_pix=45 * oversample_factor)

        # build Pandeia input
        calc = pandeia_input.build_pandeia_calc(csv=csv,
                                                array=model, 
                                                lens=lens, 
                                                band='f106', 
                                                oversample_factor=oversample_factor)

        # do Pandeia calculation        
        results, execution_time = pandeia_input.get_pandeia_results(calc)

        pandeia_output = PandeiaOutput(results)
        execution_times.append(execution_time)
        
        # save detector image
        np.save(os.path.join(array_dir, f'test_physical_lens_image_{oversample_factor}'), pandeia_output.get_image())

        # save results
        with open(os.path.join(pickle_dir, f'test_physical_lens_results_{oversample_factor}'), 'ab') as results_file:
            pickle.dump(results, results_file)

    # save list of execution times
    np.save(os.path.join(array_dir, 'test_physical_lens_execution_times.npy'), execution_times)


if __name__ == '__main__':
    main()
