import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import colors
import pickle
from glob import glob
from pprint import pprint
from tqdm import tqdm
import hydra


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):

    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util
    from mejiro.instruments.hwo import HWO
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.exposure import Exposure

    hwo = HWO()
    lens = SampleStrongLens()
    synth = SyntheticImage(lens, hwo, 'J', arcsec=5., oversample=5)
    exposure = Exposure(synth, 1000)


if __name__ == '__main__':
    main()
