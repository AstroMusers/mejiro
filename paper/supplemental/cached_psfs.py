import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util
    from mejiro.helpers import psf

    machine = HydraConfig.get().runtime.choices.machine
    if machine == 'hpc':
        os.environ['WEBBPSF_PATH'] = '/data/bwedig/STScI/webbpsf-data'
    elif machine == 'uzay':
        os.environ['WEBBPSF_PATH'] = '/data/scratch/btwedig/STScI/ref_data/webbpsf-data'

    # set directory for all output of this script
    save_dir = os.path.join(config.machine.repo_dir, 'mejiro', 'data', 'cached_psfs')
    util.create_directory_if_not_exists(save_dir)

    oversamples = [1, 3, 5]
    bands = ['F106', 'F129', 'F184']
    detectors = [4, 1, 9, 17]
    detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

    for oversample in oversamples:
        for band in bands:
            for detector, detector_position in zip(detectors, detector_positions):
                webbpsf_psf = psf.get_webbpsf_psf(band, detector, detector_position, oversample)
                psf_filename = f'{band}_{detector}_{detector_position[0]}_{detector_position[1]}_{oversample}.pkl'
                psf_path = os.path.join(save_dir, psf_filename)
                util.pickle(psf_path, webbpsf_psf)
                print(f'Pickled {psf_path}')


if __name__ == '__main__':
    main()
