import multiprocessing
import os
import random
import sys
import time
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '04'


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.instruments.roman import Roman
    from mejiro.utils import util

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    psf_cache_dir = config['psf_cache_dir']
    limit = config['limit']
    synthetic_image_config = config['synthetic_image']
    psf_config = config['psf']

    # set configuration parameters
    psf_cache_dir = os.path.join(data_dir, psf_cache_dir)
    psf_config['psf_cache_dir'] = psf_cache_dir

    # set up top directory for all pipeline output
    pipeline_dir = os.path.join(data_dir, config['pipeline_dir'])
    if dev:
        pipeline_dir += '_dev'

    # tell script where the output of previous script is
    input_dir = os.path.join(pipeline_dir, PREV_SCRIPT_NAME)
    input_sca_dirs = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in input_sca_dirs])
    scas = [str(sca).zfill(2) for sca in scas]
    if verbose: print(f'Reading from {input_sca_dirs}')

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    output_sca_dirs = []
    for sca in scas:
        sca_dir = os.path.join(output_dir, f'sca{sca}')
        os.makedirs(sca_dir, exist_ok=True)
        output_sca_dirs.append(sca_dir)
    if verbose: print(f'Set up output directories {output_sca_dirs}')

    # build instrument
    roman = Roman()

    # parse uids
    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_dir + f'/sca{sca}/lens_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[1].split('.')[0] for i in pickled_lenses]
        uid_dict[sca] = lens_uids
    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)
    if limit is not None:
        lens_uids = lens_uids[:limit]
        if limit < count:
            count = limit
    if verbose: print(f'Processing {count} lens(es)')

    # tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_sca_dir = os.path.join(input_dir, f'sca{sca}')
        output_sca_dir = os.path.join(output_dir, f'sca{sca}')

        for uid in lens_uids:
            tuple_list.append((uid, sca, roman, synthetic_image_config, psf_config, input_sca_dir, output_sca_dir))
            i += 1
            if i == limit:
                break
        else:
            continue
        break

    # Define the number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count
    process_count -= config['headroom_cores']['script_04']
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # Submit tasks to the executor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(create_synthetic_image, task): task for task in tuple_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline_dir, 'execution_times.json'))


def create_synthetic_image(input):
    from mejiro.engines.stpsf_engine import STPSFEngine
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils import roman_util, util

    # unpack tuple
    (uid, sca, roman, synthetic_image_config, psf_config, input_dir, output_dir) = input

    # unpack pipeline params
    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    pieces = synthetic_image_config['pieces']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    divide_up_sca = psf_config['divide_up_sca']
    psf_cache_dir = psf_config['psf_cache_dir']
    num_pix = psf_config['num_pixes'][0]

    # load the lens based on uid
    lens = util.unpickle(os.path.join(input_dir, f'lens_{uid}.pkl'))
    lens_uid = lens.name.split('_')[2]
    assert lens_uid == uid, f'UID mismatch: {lens_uid} != {uid}'

    # build kwargs_numerics
    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode
    }

    # set detector and pick random position
    possible_detector_positions = roman_util.divide_up_sca(divide_up_sca)
    detector_pos = random.choice(possible_detector_positions)
    instrument_params = {
        'detector': int(sca),
        'detector_position': detector_pos
    }

    # generate synthetic images
    for band in bands:
        # get PSF
        kwargs_psf = STPSFEngine.get_roman_psf_kwargs(band, int(sca), detector_pos, oversample=supersampling_factor, num_pix=num_pix, check_cache=True, psf_cache_dir=psf_cache_dir, verbose=False)

        try:
            synthetic_image = SyntheticImage(strong_lens=lens,
                                        instrument=roman,
                                        band=band,
                                        fov_arcsec=fov_arcsec,
                                        instrument_params=instrument_params,
                                        kwargs_numerics=kwargs_numerics,
                                        kwargs_psf=kwargs_psf,
                                        pieces=pieces,
                                        verbose=False)
            util.pickle(os.path.join(output_dir, f'SyntheticImage_{uid}_{band}.pkl'), synthetic_image)
        except:
            return


if __name__ == '__main__':
    main()
