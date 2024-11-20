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
    from mejiro.utils import util
    from mejiro.lenses import lens_util
    from mejiro.instruments.roman import Roman

    machine = HydraConfig.get().runtime.choices.machine
    if machine == 'hpc':
        os.environ['WEBBPSF_PATH'] = '/data/bwedig/STScI/webbpsf-data'
    elif machine == 'uzay':
        os.environ['WEBBPSF_PATH'] = '/data/scratch/btwedig/STScI/ref_data/webbpsf-data'

    # Set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'snr_per_pixel_threshold')
    util.create_directory_if_not_exists(save_dir)
    print(f'Saving output to {save_dir}')

    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    # debugging = True  # TODO TEMP

    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    detectable_lenses = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, verbose=True)
    sample_set = detectable_lenses[:120]  # TODO TEMP

    snr_thresholds = np.linspace(0.5, 5, 10)

    roman = Roman()
    zp_dict = roman.zp_dict
    zp = zp_dict['SCA01']['F129']

    arg_list = [(lens, zp, snr_thresholds) for lens in sample_set]

    # Determine the number of processes
    count = len(arg_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = 24  # Fixed process count (adjust this as needed)
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # Process the tasks with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(calc_snr, args) for args in arg_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            uid, snrs, masked_snr_arrays = future.result()
            if uid is None:
                continue
            np.save(os.path.join(save_dir, f'snrs_{uid}.npy'), snrs)
            np.save(os.path.join(save_dir, f'masked_snr_arrays_{uid}.npy'), masked_snr_arrays)

    stop = time.time()
    util.print_execution_time(start, stop)


def calc_snr(args):
    from mejiro.helpers import survey_sim

    lens, zp, snr_thresholds = args

    snrs = []
    masked_snr_arrays = []

    for snr_threshold in tqdm(snr_thresholds, leave=False):
        snr, masked_snr_array, _, _ = survey_sim.get_snr(lens, 'F129', zp, detector=1, detector_position=(2048, 2048), input_num_pix=97, output_num_pix=91, side=10.01, oversample=5, exposure_time=146, add_subhalos=False, snr_per_pixel_threshold=snr_threshold)

        if snr is None or masked_snr_array is None:
            return None, None, None
        
        masked_snr_arrays.append(masked_snr_array)
        snrs.append(snr)
    
    return lens.uid, snrs, masked_snr_arrays


if __name__ == '__main__':
    main()
