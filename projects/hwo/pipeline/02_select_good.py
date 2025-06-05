import os
import sys
import yaml
from glob import glob
from pprint import pprint

import numpy as np
import astropy.cosmology as astropy_cosmo
import pandas as pd
from dolphin.analysis.output import Output
from tqdm import tqdm


def main():
    # enable use of local modules
    repo_dir = '/grad/bwedig/mejiro'
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # set up directories to save output to
    data_dir = '/data/bwedig/mejiro'
    save_dir = os.path.join(data_dir, 'hwo', 'dinos_good')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    # set source directory
    source_dir = os.path.join(data_dir, 'hwo', 'dinos')

    # indicate the good files
    good_files = ["SDSSJ0029-0055.pkl", "SDSSJ0037-0942.pkl", "SDSSJ0151+0049.pkl", "SDSSJ0252+0039.pkl", "SDSSJ0330-0020.pkl", "SDSSJ0728+3835.pkl", "SDSSJ0737+3216.pkl", "SDSSJ0747+5055.pkl", "SDSSJ0801+4727.pkl", "SDSSJ0819+4534.pkl", "SDSSJ0830+5116.pkl", "SDSSJ0903+4116.pkl", "SDSSJ0912+0029.pkl", "SDSSJ0959+0410.pkl", "SDSSJ1023+4230.pkl", "SDSSJ1100+5329.pkl", "SDSSJ1112+0826.pkl", "SDSSJ1159-0007.pkl", "SDSSJ1204+0358.pkl", "SDSSJ1215+0047.pkl", "SDSSJ1218+0830.pkl", "SDSSJ1221+3806.pkl", "SDSSJ1234-0241.pkl", "SDSSJ1250+0523.pkl", "SDSSJ1318-0104.pkl", "SDSSJ1337+3620.pkl", "SDSSJ1349+3612.pkl", "SDSSJ1352+3216.pkl", "SDSSJ1402+6321.pkl", "SDSSJ1531-0105.pkl", "SDSSJ1542+1629.pkl", "SDSSJ1545+2748.pkl", "SDSSJ1601+2138.pkl", "SDSSJ1621+3931.pkl", "SDSSJ1627-0053.pkl", "SDSSJ1630+4520.pkl", "SDSSJ1631+1854.pkl", "SDSSJ1636+4707.pkl", "SDSSJ2125+0411.pkl", "SDSSJ2238-0754.pkl", "SDSSJ2300+0022.pkl", "SDSSJ2302-0840.pkl", "SDSSJ2303+0037.pkl", "SDSSJ2303+1422.pkl", "SL2SJ0219-0829.pkl", "SL2SJ0849-0412.pkl", "SL2SJ1405+5243.pkl", "SL2SJ1411+5651.pkl", "SL2SJ1420+5630.pkl"]

    for file in good_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(save_dir, file)
        if os.path.exists(src):
            os.system(f"cp {src} {dst}")
        else:
            print(f"File not found: {src}")

if __name__ == '__main__':
    main()
