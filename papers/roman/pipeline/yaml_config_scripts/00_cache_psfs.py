import multiprocessing
import os
import sys
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import roman_util, util
    from mejiro.engines.stpsf_engine import STPSFEngine

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    data_dir = config['data_dir']
    psf_cache_dir = config['psf_cache_dir']
    psf_config = config['psf']
    oversamples = psf_config['oversamples']
    bands = psf_config['bands']
    detectors = psf_config['detectors']
    detector_positions = roman_util.divide_up_sca(psf_config['divide_up_sca'])
    num_pixes = psf_config['num_pixes']

    # Set directory for all output of this script
    save_dir = os.path.join(data_dir, psf_cache_dir)
    util.create_directory_if_not_exists(save_dir)
    print(f'Saving PSFs to {save_dir}')

    # Determine which PSFs need to be generated
    psf_ids = []
    for oversample in oversamples:
        for num_pix in num_pixes:
            for band in bands:
                for detector in detectors:
                    for detector_position in detector_positions:
                        psf_id = STPSFEngine.get_psf_id(band, detector, detector_position, oversample, num_pix)
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
    process_count = cpu_count - config['headroom_cores']['script_00']
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s) to generate {count} PSF(s)')

    # Process the tasks with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(generate_psf, args) for args in arg_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    util.print_execution_time(start, stop)


def generate_psf(args):
    from mejiro.engines.stpsf_engine import STPSFEngine

    # Unpack tuple
    psf_id, save_dir = args

    # Generate PSF
    webbpsf_psf = STPSFEngine.get_roman_psf_from_id(psf_id, check_cache=False, verbose=False)

    # Save PSF
    psf_path = os.path.join(save_dir, f'{psf_id}.npy')
    np.save(psf_path, webbpsf_psf)


if __name__ == '__main__':
    main()
