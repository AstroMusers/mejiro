import datetime
import galsim
import hydra
import multiprocessing
import numpy as np
import os
import sys
import time
from multiprocessing import Pool
from glob import glob
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
    output_dir = config.machine.dir_04
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # list uids
    # TODO LIMIT IS TEMP
    # limit = 9
    # uid_list = list(range(limit))
    # count number of lenses and build indices of uids
    lens_pickles = glob(config.machine.dir_02 + '/lens_with_subhalos_*')
    count = len(lens_pickles)
    uid_list = list(range(count))

    # get bands
    bands = util.hydra_to_dict(config.pipeline)['band']

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4
    if count < process_count:
        process_count = count       
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for uid in uid_list:
        tuple_list.append((uid, pipeline_params, input_dir, output_dir, bands))

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
    from mejiro.helpers import gs, psf
    from mejiro.utils import util

    # unpack tuple
    (uid, pipeline_params, input_dir, output_dir, bands) = input

    # unpack pipeline_params
    grid_oversample = pipeline_params['grid_oversample']
    exposure_time = pipeline_params['exposure_time']
    suppress_output = pipeline_params['suppress_output']
    final_pixel_side = pipeline_params['final_pixel_side']
    num_pix = pipeline_params['num_pix']

    # load lens
    lens = util.unpickle(os.path.join(input_dir, f'lens_{str(uid).zfill(8)}'))

    # create galsim rng
    rng = galsim.UniformDeviate()

    # determine detector and position
    detector = gs.get_random_detector(suppress_output)
    detector_pos = psf.get_random_detector_pos(input_size=num_pix, suppress_output=suppress_output)

    # get wcs
    wcs_dict = gs.get_random_hlwas_wcs(suppress_output)

    # calculate sky backgrounds for each band
    bkgs = gs.get_sky_bkgs(wcs_dict, bands, detector, exposure_time, num_pix=num_pix)

    execution_times = []

    # TODO fix this loop once gs.py is finalized - it should only be a few lines because can save image for each band with list comprehension
    for _, band in enumerate(bands):
        start = time.time()      

        # load the appropriate array
        array = np.load(f'{input_dir}/array_{lens.uid}_{band}.npy')

        # get flux
        total_flux_cps = lens.get_total_flux_cps(band)  
        
        # get interpolated image
        interp = galsim.InterpolatedImage(galsim.Image(array, xmin=0, ymin=0), scale=0.11 / grid_oversample, flux=total_flux_cps * exposure_time)

        # generate PSF and convolve
        convolved = gs.convolve(interp, band, detector, detector_pos, num_pix, pupil_bin=1)

        # add sky background to convolved image
        final_image = convolved + bkgs[band]

        # integer number of photons are being detected, so quantize
        final_image.quantize()

        # add all detector effects
        galsim.roman.allDetectorEffects(final_image, prev_exposures=(), rng=rng, exptime=exposure_time)

        # make sure there are no negative values from Poisson noise generator
        final_image.replaceNegative()

        # get the array
        final_array = final_image.array

        # center crop to get rid of edge effects
        final_array = util.center_crop_image(final_array, (final_pixel_side, final_pixel_side))

        # divide through by exposure time to get in units of counts/sec/pixel
        final_array /= exposure_time

        np.save(os.path.join(output_dir, f'galsim_{lens.uid}_{band}.npy'), final_array)

        stop = time.time()
        execution_time = str(datetime.timedelta(seconds=round(stop - start)))
        execution_times.append(execution_time)

    return execution_times


if __name__ == '__main__':
    main()
