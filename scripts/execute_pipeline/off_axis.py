import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import hydra
import numpy as np
from astropy import convolution
from skimage import restoration
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # directory to write the output to
    output_dir = os.path.join(config.machine.pipeline_dir, '04_test')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # directory where default PSFs live
    psf_dir = os.path.join(repo_dir, 'mejiro', 'data', 'default_psfs')

    # open pickled lens dict list
    limit = 25
    f106_list = util.unpickle_all(config.machine.dir_03, prefix='lens_dict_*_f106', limit=limit)
    f129_list = util.unpickle_all(config.machine.dir_03, prefix='lens_dict_*_f129', limit=limit)
    f184_list = util.unpickle_all(config.machine.dir_03, prefix='lens_dict_*_f184', limit=limit)
    dict_list = []
    for i, _ in enumerate(f106_list):
        dict_list.append((f106_list[i], f129_list[i], f184_list[i]))
    # dict_list = util.unpickle_all(config.machine.dir_03)

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    # process_count = cpu_count - 4
    # if len(dict_list) < process_count:
    #     process_count = len(dict_list)
    process_count = 13
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for i, _ in enumerate(dict_list):
        tuple_list.append((dict_list[i], pipeline_params, output_dir, psf_dir))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(get_image, batch)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image(input):
    from mejiro.helpers import pandeia_input, psf

    # unpack tuples
    (lens_dict_tuple, pipeline_params, output_dir, psf_dir) = input

    # unpack pipeline_params
    max_scene_size = pipeline_params['max_scene_size']
    num_samples = pipeline_params['num_samples']
    oversample = pipeline_params['grid_oversample']

    # get parameters for off-axis PSF
    instrument = psf.get_instrument('F106')
    x, y = psf.get_random_position(suppress_output=False)
    detector = psf.get_random_detector(instrument, suppress_output=False)

    for lens_dict in lens_dict_tuple:
        # unpack lens_dict
        array = lens_dict['model']
        lens = lens_dict['lens']
        uid = lens.uid
        band = lens.band

        print(f'Lens {uid}, band {band}')

        # generate Pandeia image with no background or noise
        calc_off, _ = pandeia_input.build_pandeia_calc(array,
                                                       lens,
                                                       background=False,
                                                       noise=False,
                                                       band=band,
                                                       max_scene_size=max_scene_size,
                                                       num_samples=num_samples,
                                                       suppress_output=False)
        pandeia_off, _ = pandeia_input.get_pandeia_image(calc_off, suppress_output=False)
        pandeia_off = np.nan_to_num(pandeia_off, copy=False, nan=0)

        # generate Pandeia image with background and noise
        calc_on, _ = pandeia_input.build_pandeia_calc(array,
                                                      lens,
                                                      background=True,
                                                      noise=True,
                                                      band=band,
                                                      max_scene_size=max_scene_size,
                                                      num_samples=num_samples,
                                                      suppress_output=False)
        pandeia_on, _ = pandeia_input.get_pandeia_image(calc_on, suppress_output=False)

        # subtract to get noise and convolved sky background
        noise_and_convolved_bkg = pandeia_on - pandeia_off

        # deconvolve the "nothing on" image to get synthetic image in Pandeia units
        default_kernel = psf.load_default_psf(psf_dir, band, oversample)
        deconvolved = restoration.richardson_lucy(pandeia_off,
                                                  default_kernel,
                                                  num_iter=30,
                                                  clip=False)

        # re-convolve with off-axis PSF
        off_axis_kernel = psf.get_psf_kernel(band=band,
                                             detector=detector,
                                             detector_position=(x, y),
                                             oversample=oversample,
                                             suppress_output=False)
        off_axis_image = convolution.convolve(deconvolved, off_axis_kernel)

        # add off-axis image and noise+convolved bkg
        final_image = off_axis_image + noise_and_convolved_bkg
        np.save(os.path.join(output_dir, f'pandeia_{uid}_{band}.npy'), final_image)

        # also save residual
        residual = final_image - pandeia_on
        np.save(os.path.join(output_dir, f'residual_{uid}_{band}.npy'), residual)


if __name__ == '__main__':
    main()
