import hydra
import json
import multiprocessing
import numpy as np
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # Set up environment variables based on the machine
    machine = HydraConfig.get().runtime.choices.machine
    if machine == 'hpc':
        os.environ['WEBBPSF_PATH'] = '/data/bwedig/STScI/webbpsf-data'
    elif machine == 'uzay':
        os.environ['WEBBPSF_PATH'] = '/data/scratch/btwedig/STScI/ref_data/webbpsf-data'

    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir

    # Enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util

    # Retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    limit = pipeline_params['limit']

    # Set nice level
    os.nice(pipeline_params['nice'])

    # Set up input directories
    if debugging:
        input_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '03')
    else:
        input_parent_dir = config.machine.dir_03
    sca_dirnames = [os.path.basename(d) for d in glob(os.path.join(input_parent_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dirnames])
    scas = [str(sca).zfill(2) for sca in scas]

    # Set up output directories
    if debugging:
        output_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '04')
    else:
        output_parent_dir = config.machine.dir_04
    util.create_directory_if_not_exists(output_parent_dir)
    util.clear_directory(output_parent_dir)
    for sca in scas:
        os.makedirs(os.path.join(output_parent_dir, f'sca{sca}'), exist_ok=True)

    # Set PSF cache dir
    psf_cache_dir = os.path.join(config.machine.data_dir, 'cached_psfs')

    # Get lens UIDs, organized by SCA
    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_parent_dir + f'/sca{sca}/array_*.npy'))
        lens_uids = [os.path.basename(i).split('_')[1] for i in pickled_lenses]
        lens_uids = list(set(lens_uids))  # Remove duplicates
        lens_uids = sorted(lens_uids)
        uid_dict[sca] = lens_uids

    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)

    if limit != 'None' and limit < count:
        count = limit

    # Get zeropoint magnitudes
    module_path = os.path.dirname(mejiro.__file__)
    zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

    # Determine number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)
    process_count -= 4  # uzay needs even more headroom
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # Tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_dir = os.path.join(input_parent_dir, f'sca{sca}')
        output_dir = os.path.join(output_parent_dir, f'sca{sca}')
        sca_zp_dict = zp_dict[f'SCA{sca}']

        for uid in lens_uids:
            tuple_list.append((uid, sca, pipeline_params, sca_zp_dict, input_dir, output_dir, psf_cache_dir))
            i += 1
            if i == limit:
                break
        else:
            continue
        break

    # Process the tasks with ProcessPoolExecutor
    execution_times = []
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(get_image, task) for task in tuple_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            execution_times.extend(future.result())

    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '04',
                              os.path.join(os.path.dirname(output_parent_dir), 'execution_times.json'))


def get_image(input):
    from mejiro.helpers import gs
    from mejiro.utils import roman_util, util

    # Unpack tuple
    (uid, sca, pipeline_params, sca_zp_dict, input_dir, output_dir, psf_cache_dir) = input

    # Unpack pipeline_params
    bands = pipeline_params['bands']
    grid_oversample = pipeline_params['grid_oversample']
    exposure_time = pipeline_params['exposure_time']
    suppress_output = pipeline_params['suppress_output']
    final_pixel_side = pipeline_params['final_pixel_side']
    num_pix = pipeline_params['num_pix']
    pieces = pipeline_params['pieces']

    # Load lens
    try:
        lens = util.unpickle(os.path.join(input_dir, f'lens_{uid}.pkl'))
    except Exception as e:
        print(f'Error unpickling lens {uid}: {e}')
        return 0

    # Load the appropriate arrays
    arrays = [np.load(f'{input_dir}/array_{lens.uid}_{band}.npy') for band in bands]
    if pieces:
        lens_surface_brightness = [np.load(f'{input_dir}/array_{lens.uid}_lens_{band}.npy') for band in bands]
        source_surface_brightness = [np.load(f'{input_dir}/array_{lens.uid}_source_{band}.npy') for band in bands]
        pieces_args = {'lens_surface_brightness': lens_surface_brightness,
                       'source_surface_brightness': source_surface_brightness}
    else:
        pieces_args = {}

    # Determine detector and position
    detector = int(sca)
    possible_detector_positions = roman_util.divide_up_sca(5)
    detector_pos = random.choice(possible_detector_positions)

    # Save attributes on StrongLens
    lens.detector = detector
    lens.detector_position = detector_pos
    util.pickle(os.path.join(input_dir, f'lens_{uid}.pkl'), lens)

    gs_results = gs.get_images(lens,
                               arrays,
                               bands,
                               sca_zp_dict,
                               input_size=num_pix,
                               output_size=final_pixel_side,
                               grid_oversample=grid_oversample,
                               psf_oversample=grid_oversample,
                               **pieces_args,
                               detector=detector,
                               detector_pos=detector_pos,
                               exposure_time=exposure_time,
                               ra=None,
                               dec=None,
                               seed=random.randint(0, 2 ** 16 - 1),
                               validate=False,
                               psf_cache_dir=psf_cache_dir,
                               suppress_output=suppress_output)

    if pieces:
        results, lenses, sources, execution_time = gs_results
        results += lenses
        results += sources
        bands *= 3  # Repeat bands 3 times for pieces
    else:
        results, execution_time = gs_results

    j = 0
    for i, (band, result) in enumerate(zip(bands, results)):
        if j == 0:
            out_path = os.path.join(output_dir, f'galsim_{lens.uid}_{band}.npy')
            if not suppress_output:
                print(f'Writing {out_path}...')
            np.save(out_path, result)
        elif j == 1:
            out_path = os.path.join(output_dir, f'galsim_{lens.uid}_lens_{band}.npy')
            if not suppress_output:
                print(f'Writing {out_path}...')
            np.save(out_path, result)
        elif j == 2:
            out_path = os.path.join(output_dir, f'galsim_{lens.uid}_source_{band}.npy')
            if not suppress_output:
                print(f'Writing {out_path}...')
            np.save(out_path, result)
        if i % len(set(bands)) == len(set(bands)) - 1:
            j += 1

    return execution_time


if __name__ == '__main__':
    main()
