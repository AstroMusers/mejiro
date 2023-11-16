import os

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'

from astropy.io import fits
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize
from astropy.wcs import WCS



