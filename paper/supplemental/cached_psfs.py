import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util
    from mejiro.helpers import psf
    from mejiro.instruments.roman import Roman

    machine = HydraConfig.get().runtime.choices.machine
    if machine == 'hpc':
        os.environ['WEBBPSF_PATH'] = '/data/bwedig/STScI/webbpsf-data'
    elif machine == 'uzay':
        os.environ['WEBBPSF_PATH'] = '/data/scratch/btwedig/STScI/ref_data/webbpsf-data'

    # set directory for all output of this script
    # save_dir = os.path.join(config.machine.repo_dir, 'mejiro', 'data', 'cached_psfs')
    save_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    util.create_directory_if_not_exists(save_dir)

    oversamples = [1, 3, 5]
    # bands = ['F106', 'F129', 'F158', 'F184']
    bands = ['F087']
    # detectors = [4, 1, 9, 17]
    # detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]
    # detectors = [1, 2, 4, 5]
    # detector_positions = [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048)]
    detectors = list(range(1, 19))
    # detector_positions = [(2048, 2048)] * 18
    detector_positions = []
    for i in range(4):
        detector_positions.extend(Roman().divide_up_sca(i + 1))
    num_pixes = [47, 141, 235]

    num_iterations = len(oversamples) * len(bands) * len(detectors) * len(detector_positions)
    for oversample, num_pix in zip(oversamples, num_pixes):
        for band in bands:
            for detector in tqdm(detectors, leave=True):
                for detector_position in tqdm(detector_positions, leave=False):
                    psf_filename = f'{psf.get_psf_id_string(band, detector, detector_position, oversample, num_pix)}.pkl'
                    psf_path = os.path.join(save_dir, psf_filename)
                    if os.path.exists(psf_path):
                        print(f'{psf_path} already exists')
                        continue
                    else:
                        webbpsf_psf = psf.get_webbpsf_psf(band, detector, detector_position, oversample, num_pix)
                        util.pickle(psf_path, webbpsf_psf)
                        print(f'Pickled {psf_path}')


if __name__ == '__main__':
    main()
