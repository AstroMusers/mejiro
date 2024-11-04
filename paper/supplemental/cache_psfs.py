import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import hydra
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    os.nice(19)

    # Enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import roman_util, util
    from mejiro.engines import webbpsf_engine
    from mejiro.instruments.roman import Roman

    machine = HydraConfig.get().runtime.choices.machine
    if machine == 'hpc':
        os.environ['WEBBPSF_PATH'] = '/data/bwedig/STScI/webbpsf-data'
    elif machine == 'uzay':
        os.environ['WEBBPSF_PATH'] = '/data/scratch/btwedig/STScI/ref_data/webbpsf-data'

    # Set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    util.create_directory_if_not_exists(save_dir)
    print(f'Saving PSFs to {save_dir}')

    oversamples = [5]
    bands = ['F087']  # , 'F106', 'F129', 'F158', 'F184'
    # bands = Roman().bands
    # detectors = [4, 1, 9, 17]
    # detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]
    # detectors = [1, 2, 4, 5]
    # detector_positions = [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048)]
    detectors = list(range(1, 19))
    # detector_positions = [(2048, 2048)]
    # detector_positions = []
    # for i in range(4):
    #     detector_positions.extend(roman_util.divide_up_sca(i + 1))
    detector_positions = roman_util.divide_up_sca(3)
    num_pixes = [101]

    # Determine which PSFs need to be generated
    psf_ids = []
    for oversample, num_pix in zip(oversamples, num_pixes):
        for band in bands:
            for detector in detectors:
                for detector_position in detector_positions:
                    psf_id = webbpsf_engine.get_psf_id(band, detector, detector_position, oversample, num_pix)
                    psf_filename = f'{psf_id}.npy'
                    psf_path = os.path.join(save_dir, psf_filename)
                    if os.path.exists(psf_path):
                        print(f'{psf_path} already exists')
                        continue
                    else:
                        psf_ids.append(psf_id)

    if len(psf_ids) == 0:
        print('All PSFs already exist. Exiting.')
        return

    arg_list = [(psf_id_string, save_dir) for psf_id_string in psf_ids]

    # Determine the number of processes
    count = len(arg_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = 24  # Fixed process count (you can adjust this as needed)
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s) to generate {count} PSF(s)')

    # Process the tasks with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(generate_psf, args) for args in arg_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    util.print_execution_time(start, stop)


def generate_psf(args):
    from mejiro.engines import webbpsf_engine

    # Unpack tuple
    psf_id, save_dir = args

    # Generate PSF
    webbpsf_psf = webbpsf_engine.get_roman_psf_from_id(psf_id, check_cache=False, verbose=False)

    # Save PSF
    psf_path = os.path.join(save_dir, f'{psf_id}.npy')
    np.save(psf_path, webbpsf_psf)
    # Print statement optional for debugging purposes
    # print(f'Saved {psf_path}')


if __name__ == '__main__':
    main()
