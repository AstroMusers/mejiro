import os
import sys

import hydra
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util
    from mejiro.helpers import psf

    # set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    util.create_directory_if_not_exists(save_dir)

    oversamples = [5]
    bands = ['F106', 'F129', 'F158', 'F184']
    detectors = list(range(1, 19))

    side = int(4088 / 4)
    detector_positions = [(side, side), (3 * side, side), (side, 3 * side), (3 * side, 3 * side), (side * 2, side * 2)]

    num_scas = 18
    num_points = len(detector_positions)
    num_filters = 4

    print(f'Total PSFs: {num_scas * num_points * num_filters}')

    for oversample in oversamples:
        for band in tqdm(bands, desc='Bands', leave=True):
            for detector in tqdm(detectors, desc='Detectors', leave=False):
                for detector_position in tqdm(detector_positions, desc='Detector Positions', leave=False):
                    webbpsf_psf = psf.get_webbpsf_psf(band, detector, detector_position, oversample)
                    psf_filename = f'{band}_{detector}_{detector_position[0]}_{detector_position[1]}_{oversample}.pkl'
                    psf_path = os.path.join(save_dir, psf_filename)
                    util.pickle(psf_path, webbpsf_psf)
                    # print(f'Pickled {psf_path}')


if __name__ == '__main__':
    main()
