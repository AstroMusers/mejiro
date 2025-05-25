import galsim
import getpass
import h5py
import lenstronomy
import numpy as np
import os
import platform
import sys
import time
import yaml
import stpsf
from datetime import datetime
from glob import glob
from tqdm import tqdm


PREV_SCRIPT_NAME = '05'
SCRIPT_NAME = '06'


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    limit = config['limit']
    bands = config['synthetic_image']['bands']
    imaging_config = config['imaging']
    exposure_time = imaging_config['exposure_time']
    snr_band = config['snr']['snr_band']

    # set up top directory for all pipeline output
    pipeline_dir = os.path.join(data_dir, config['pipeline_dir'])
    if dev:
        pipeline_dir += '_dev'

    # tell script where the output of previous script is
    input_dir = os.path.join(pipeline_dir, '05')

    # parse output of previous script to determine which SCAs to process
    sca_dir_names = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dir_names])
    scas = [str(sca).zfill(2) for sca in scas]

    # get lens UIDs, organized by SCA
    uid_dict = {}
    for sca in scas:
        pickled_exposures = sorted(glob(input_dir + f'/sca{sca}/Exposure_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[1] for i in pickled_exposures]
        lens_uids = list(set(lens_uids))  # remove duplicates, e.g., if multiple bands
        lens_uids = sorted(lens_uids)
        uid_dict[sca] = lens_uids  # for each SCA, list of UIDs of associated lenses

    if verbose: print(f'Found {len(uid_dict)} SCA(s) with {sum([len(v) for v in uid_dict.values()])} lenses')

    # create h5 file
    filepath = f'{pipeline_dir}/roman_data_v1.h5'
    if os.path.exists(filepath):
        if verbose: print(f'File {filepath} already exists. Overwriting...')
        os.remove(filepath)
    f = h5py.File(filepath, 'a')  # append mode: read/write if exists, create otherwise

    # set file-level attributes
    f.attrs['author'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    f.attrs['created'] = (now_string)
    f.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    f.attrs['lenstronomy_version'] = (lenstronomy.__version__, 'lenstronomy version')
    f.attrs['galsim_version'] = (galsim.__version__, 'GalSim version')
    f.attrs['stpsf_version'] = (stpsf.__version__, 'STPSF version')

    # ---------------------------CREATE IMAGE DATASET--------------------------------
    for band in tqdm(bands, desc='Bands', position=1, leave=True):
        group_band = f.create_group(band)

        for sca, uid_list in tqdm(uid_dict.items(), desc=f'SCAs', position=2, leave=False):
            for uid in tqdm(uid_list, desc='Strong Lenses', position=3, leave=False):
                exposure_pickle = sorted(glob(input_dir + f'/sca{sca}/Exposure_{uid}_{band}.pkl'))
                if len(exposure_pickle) != 1:
                    raise ValueError(f'Found {len(exposure_pickle)} pickled exposures for UID {uid} in SCA {sca}. Expected 1.')

                # grab an exposure and strong lens
                exposure = util.unpickle(exposure_pickle[0])
                lens = exposure.synthetic_image.strong_lens

                for is_strong_lens in [True, False]:
                    # create datasets
                    if is_strong_lens:
                        dset = group_band.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}_{is_strong_lens}', data=exposure.exposure)
                        dset.attrs['is_strong_lens'] = True
                    else:
                        lens_with_noise = exposure.exposure - exposure.source_exposure
                        dset = group_band.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}_{is_strong_lens}', data=lens_with_noise)
                        dset.attrs['is_strong_lens'] = False


                    dset.attrs['units'] = ('Counts/sec', 'Units of pixel values')
                    dset.attrs['filter'] = (band, 'Filter')
                    dset.attrs['source_magnitude'] = (str(lens.physical_params['magnitudes']['source'][band]), 'Unlensed source galaxy magnitude')
                    dset.attrs['lensed_source_magnitude'] = (
                        str(lens.physical_params['magnitudes']['lensed_source'][band]), 'Lensed source galaxy magnitude')
                    dset.attrs['lens_magnitude'] = (str(lens.physical_params['magnitudes']['lens'][band]), 'Lens galaxy magnitude')
                    dset.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
                    dset.attrs['name'] = (lens.name, 'Unique identifier for system assigned by mejiro')
                    dset.attrs['z_source'] = (str(lens.z_source), 'Source galaxy redshift')
                    dset.attrs['z_lens'] = (str(lens.z_lens), 'Lens galaxy redshift')
                    dset.attrs['theta_e'] = (str(lens.get_einstein_radius()), 'Einstein radius [arcsec]')
                    dset.attrs['sigma_v'] = (str(lens.physical_params.get('lens_vel_disp')), 'Lens galaxy velocity dispersion [km/s]')
                    dset.attrs['mu'] = (str(lens.physical_params.get('magnification')), 'Flux-weighted magnification of source')
                    dset.attrs['instrument'] = ('WFI', 'Instrument')
                    dset.attrs['exposure_time'] = (str(exposure_time), 'Exposure time [seconds]')
                    dset.attrs['pixel_scale'] = (str(exposure.synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
                    dset.attrs['fov'] = (str(exposure.synthetic_image.fov_arcsec), 'Field of view [arcsec]')
                    dset.attrs['detector'] = (str(exposure.synthetic_image.instrument_params.get('detector')), 'Detector')
                    dset.attrs['detector_position_x'] = (str(exposure.synthetic_image.instrument_params.get('detector_position')[0]), 'Detector X position')
                    dset.attrs['detector_position_y'] = (str(exposure.synthetic_image.instrument_params.get('detector_position')[1]), 'Detector Y position')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
