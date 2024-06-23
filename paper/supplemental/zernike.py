import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from glob import glob
from pprint import pprint
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
import hydra


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util

    # set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'output', 'zernikes')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    # read cycle 9 Zernicke coefficients CSV
    import mejiro
    module_path = os.path.dirname(mejiro.__file__)
    zernike_csv_path = os.path.join(module_path, 'data', 'wim_zernikes_cycle9.csv')
    df = pd.read_csv(zernike_csv_path)

    # set wavelength
    wavelength = 1.06

    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)

    # loop through Zernicke coefficients
    zernikes = []
    for z in tqdm(range(1, 23)):
        scas = []

        # loop through SCAs
        for sca in range(1, 19):
            sca_df = df[df['sca'] == sca]
            sca_df = sca_df[sca_df['wavelength'] == wavelength]

            points, values = [], []
            for _, row in sca_df.iterrows():
                points.append([row['local_x'], row['local_y']])
                values.append(row[f'Z{z}'])

            grid_z = griddata(points, values, (X, Y), method='cubic')
            grid_z = np.flipud(grid_z)

            scas.append(grid_z)

        zernikes.append(scas)

    np.save(os.path.join(save_dir, 'zernikes.npy'), zernikes)


if __name__ == '__main__':
    main()
