import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import hydra
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    os.nice(19)

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import roman_util, util
    from mejiro.helpers import psf

    machine = HydraConfig.get().runtime.choices.machine
    if machine == 'hpc':
        os.environ['WEBBPSF_PATH'] = '/data/bwedig/STScI/webbpsf-data'
    elif machine == 'uzay':
        os.environ['WEBBPSF_PATH'] = '/data/scratch/btwedig/STScI/ref_data/webbpsf-data'

    # set directory for all output of this script
    # save_dir = os.path.join(config.machine.repo_dir, 'mejiro', 'data', 'cached_psfs')
    save_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    util.create_directory_if_not_exists(save_dir)

    oversamples = [1]
    bands = ['F087']  # , 'F106', 'F129', 'F158', 'F184'
    # bands = ['F129']
    # detectors = [4, 1, 9, 17]
    # detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]
    # detectors = [1, 2, 4, 5]
    # detector_positions = [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048)]
    detectors = list(range(1, 19))
    # detector_positions = [(2048, 2048)]
    # detector_positions = []
    # for i in range(4):
    #     detector_positions.extend(roman_util.divide_up_sca(i + 1))
    detector_positions = roman_util.divide_up_sca(2)
    num_pixes = [101]

    # determine which PSFs need to be generated
    psf_id_strings = []
    for oversample, num_pix in zip(oversamples, num_pixes):
        for band in bands:
            for detector in detectors:
                for detector_position in detector_positions:
                    psf_id_string = psf.get_psf_id_string(band, detector, detector_position, oversample, num_pix)
                    psf_filename = f'{psf_id_string}.pkl'
                    psf_path = os.path.join(save_dir, psf_filename)
                    if os.path.exists(psf_path):
                        print(f'{psf_path} already exists')
                        continue
                    else:
                        psf_id_strings.append(psf_id_string)

    if len(psf_id_strings) == 0:
        print('All PSFs already exist. Exiting.')
        return

    arg_list = [(psf_id_string, save_dir) for psf_id_string in psf_id_strings]

    # split up the lenses into batches based on core count
    count = len(arg_list)
    cpu_count = multiprocessing.cpu_count()
    # process_count = cpu_count - config.machine.headroom_cores
    # if count < process_count:
    #     process_count = count
    process_count = 10
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s) to generate {count} PSF(s)')

    # batch
    generator = util.batch_list(arg_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(generate_psf, batch)

    stop = time.time()
    util.print_execution_time(start, stop)


def generate_psf(tuple):
    from mejiro.utils import util
    from mejiro.helpers import psf

    # unpack tuple
    (psf_id_string, save_dir) = tuple

    # generate PSF
    webbpsf_psf = psf.get_webbpsf_psf_from_string(psf_id_string)

    # pickle PSF
    psf_path = os.path.join(save_dir, f'{psf_id_string}.pkl')
    util.pickle(psf_path, webbpsf_psf)
    # print(f'Pickled {psf_path}')


if __name__ == '__main__':
    main()
