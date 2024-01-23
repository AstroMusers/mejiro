import hydra
import multiprocessing
import numpy as np
import os
import sys
import time
from multiprocessing import Pool
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # directory to read from
    input_dir = config.machine.dir_03

    # directory to write the output to
    output_dir = os.path.join(config.machine.pipeline_dir, '04_test')  # config.machine.dir_04
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # open pickled lens list
    # TODO LIMIT IS TEMP
    limit = 25
    lens_list = util.unpickle_all(input_dir, 'lens_', limit)

    # get bands
    bands = util.hydra_to_dict(config.pipeline)['band']

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4
    if limit < process_count:
        process_count = limit       
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for lens in lens_list:
        tuple_list.append((lens, pipeline_params, input_dir, output_dir, bands))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    execution_times = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_image, batch):
            execution_times.extend(output)

    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image(input):
    from mejiro.helpers import pandeia_input, bkg
    from mejiro.utils import util

    # unpack tuple
    (lens, pipeline_params, input_dir, output_dir, bands) = input

    # unpack pipeline_params
    grid_oversample = pipeline_params['grid_oversample']
    max_scene_size = pipeline_params['max_scene_size']
    num_samples = pipeline_params['num_samples']

    # load an array to get its shape
    num_pix, _ = np.load(f'{input_dir}/array_{lens.uid}_{bands[0]}.npy').shape

    # generate sky background
    bkgs = bkg.get_high_galactic_lat_bkg((num_pix, num_pix), bands, seed=None)
    reshaped_bkgs = [util.resize_with_pixels_centered(i, grid_oversample) for i in bkgs]

    execution_times = []
    for i, band in enumerate(bands):
        # load the appropriate array
        array = np.load(f'{input_dir}/array_{lens.uid}_{band}.npy')

        # build Pandeia input
        calc, _ = pandeia_input.build_pandeia_calc(array, lens, background=reshaped_bkgs[i], noise=True, band=band,
                                                max_scene_size=max_scene_size,
                                                num_samples=num_samples, suppress_output=True)

        # generate Pandeia image and save
        image, execution_time = pandeia_input.get_pandeia_image(calc, suppress_output=True)
        np.save(os.path.join(output_dir, f'pandeia_{lens.uid}_{band}.npy'), image)

        execution_times.append(execution_time)

    return execution_times


if __name__ == '__main__':
    main()
