import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hydra import initialize, compose

# set paths to various directories based on the machine this code is being executed on
with initialize(version_base=None, config_path='../../config'):
    config = compose(config_name='config.yaml')  # overrides=['machine=uzay']

array_dir, data_dir, figure_dir, pickle_dir, repo_dir = config.machine.array_dir, config.machine.data_dir, config.machine.figure_dir, config.machine.pickle_dir, config.machine.repo_dir

# enable use of local modules
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

# set matplotlib style
plt.style.use(f'{repo_dir}/mejiro/mplstyle/science.mplstyle')

from mejiro.plots import plot_util
from mejiro.utils import util
from mejiro.analysis import stats

array_dir = os.path.join(array_dir, 'pandeia_noise_residuals')
util.create_directory_if_not_exists(array_dir)
pickle_dir = os.path.join(pickle_dir, 'pyhalo')

all_on = np.load(os.path.join(array_dir, 'all_on.npy'))
crs_off = np.load(os.path.join(array_dir, 'crs_off.npy'))
dark_off = np.load(os.path.join(array_dir, 'dark_off.npy'))
ffnoise_off = np.load(os.path.join(array_dir, 'ffnoise_off.npy'))
readnoise_off = np.load(os.path.join(array_dir, 'readnoise_off.npy'))
scatter_off = np.load(os.path.join(array_dir, 'scatter_off.npy'))
saturation_off = np.load(os.path.join(array_dir, 'saturation_off.npy'))

fontsize = 24
matplotlib.rcParams.update({'font.size': fontsize})

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

array_list = [all_on - crs_off, all_on - dark_off, all_on - ffnoise_off, all_on - readnoise_off]
title_list = ['Cosmic ray noise', 'Dark current noise', 'Flat-field noise', 'Read noise']

norm = plot_util.get_norm(array_list, linear_width=0.00001)

side = 2

for i, array in enumerate(array_list):
    axis = ax[i // side, i % side].imshow(array, cmap='bwr', norm=norm)
    ax[i // side, i % side].set_title(title_list[i])
    ax[i // side, i % side].set_axis_off()

fig.colorbar(axis, ax=ax, ticks=[-0.1, -0.01, -0.001, -0.0001, 0, 0.0001, 0.001, 0.01, 0.1])

chi_square_list = []
chi_square_list.append(stats.chi_square(crs_off, all_on))
chi_square_list.append(stats.chi_square(dark_off, all_on))
chi_square_list.append(stats.chi_square(ffnoise_off, all_on))
chi_square_list.append(stats.chi_square(readnoise_off, all_on))
chi_square_list = ['$\chi^2 = $' + util.scientific_notation_string(i) for i in chi_square_list]

# create text boxes
props = dict(boxstyle='round', facecolor='w', alpha=0.8)
for i, title in enumerate(chi_square_list):
    ax[i // side, i % side].text(0.05, 0.95, title, transform=ax[i // side, i % side].transAxes, fontsize=fontsize,
                                 verticalalignment='top', bbox=props)

plt.savefig(os.path.join(figure_dir, 'noise_asinh_2by2.png'))
