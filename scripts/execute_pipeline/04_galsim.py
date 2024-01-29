import datetime
import galsim
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
    from mejiro.helpers import bkg, gs
    from mejiro.utils import util

    # unpack tuple
    (lens, pipeline_params, input_dir, output_dir, bands) = input

    # unpack pipeline_params
    grid_oversample = pipeline_params['grid_oversample']
    exposure_time = pipeline_params['exposure_time']

    # create galsim rng
    rng = galsim.UniformDeviate()

    # generate sky background


    execution_times = []
    for i, band in enumerate(bands):
        start = time.time()      

        # load the appropriate array
        array = np.load(f'{input_dir}/array_{lens.uid}_{band}.npy')

        # get flux
        total_flux_cps = lens.get_total_flux_cps(band)  
        
        # get interpolated image
        interp = galsim.InterpolatedImage(galsim.Image(array), scale=0.11 / grid_oversample, flux=total_flux_cps * exposure_time)

        # generate PSF and convolve
        convolved = gs.convolve()
        
        # add sky background
        sky_bkg = gs.get_sky_bkg()

        # add sky background to convolved image
        final_image = convolved + sky_bkg

        # integer number of photons are being detected, so quantize
        final_image.quantize()

        # add all detector effects
        galsim.roman.allDetectorEffects(final_image, prev_exposures=(), rng=rng, exptime=exposure_time)

        # quantize, as the analog-to-digital converter reads in an integer value
        final_image.quantize()

        # this has float values, so convert to integer values
        final_image = galsim.Image(final_image, dtype=int).array

        np.save(os.path.join(output_dir, f'galsim_{lens.uid}_{band}.npy'), final_image)

        stop = time.time()
        execution_time = str(datetime.timedelta(seconds=round(stop - start)))
        execution_times.append(execution_time)

    return execution_times


if __name__ == '__main__':
    main()
