import json
import multiprocessing
import os
import sys
import time
from glob import glob
from multiprocessing import Pool

import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # get directories
    repo_dir = config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    limit = pipeline_params['limit']

    # set nice level
    os.nice(pipeline_params['nice'])

    # directories to get pickled lenses (with subhalos) from
    if debugging:
        input_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '02')
    else:
        input_parent_dir = config.machine.dir_02
    sca_dirnames = [os.path.basename(d) for d in glob(os.path.join(input_parent_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dirnames])
    scas = [str(sca).zfill(2) for sca in scas]

    # get zeropoint magnitudes
    module_path = os.path.dirname(mejiro.__file__)
    zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

    # directories to write the output to
    if debugging:
        output_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '03')
    else:
        output_parent_dir = config.machine.dir_03
    util.create_directory_if_not_exists(output_parent_dir)
    util.clear_directory(output_parent_dir)
    for sca in scas:
        os.makedirs(os.path.join(output_parent_dir, f'sca{sca}'), exist_ok=True)

    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_parent_dir + f'/sca{sca}/lens_with_subhalos_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[3].split('.')[0] for i in pickled_lenses]
        uid_dict[sca] = lens_uids

    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)

    if limit != 'None' and limit < count:
        count = limit

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= 10  # TODO having resource issues here as well
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_dir = os.path.join(input_parent_dir, f'sca{sca}')
        output_dir = os.path.join(output_parent_dir, f'sca{sca}')
        sca_zp_dict = zp_dict[f'SCA{sca}']

        for uid in lens_uids:
            tuple_list.append((uid, pipeline_params, sca_zp_dict, input_dir, output_dir))
            i += 1
            if i == limit:
                break
        else:
            continue
        break

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):  # TODO tqdm thinks that there are all lenses, even when a limit is set
        pool = Pool(processes=process_count)
        pool.map(get_model, batch)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '03',
                              os.path.join(os.path.dirname(output_parent_dir), 'execution_times.json'))


def get_model(input):
    from mejiro.utils import util

    # unpack tuple
    (uid, pipeline_params, sca_zp_dict, input_dir, output_dir) = input

    # unpack pipeline params
    bands = pipeline_params['bands']
    num_pix = pipeline_params['num_pix']
    side = pipeline_params['side']
    grid_oversample = pipeline_params['grid_oversample']
    pieces = pipeline_params['pieces']
    supersampling_factor = pipeline_params['supersampling_factor']
    supersampling_compute_mode = pipeline_params['supersampling_compute_mode']
    supersampling_radius = pipeline_params['supersampling_radius']

    # build kwargs_numerics
    try:
        radius = supersampling_radius / (0.11 / grid_oversample)  # convert radius from arcsec to pixels
        supersampling_indices = util.create_centered_circle(N=num_pix * grid_oversample, radius=radius)
        kwargs_numerics = {
            'supersampling_factor': supersampling_factor,
            'compute_mode': supersampling_compute_mode,
            'supersampled_indexes': supersampling_indices
        }
    except Exception as e:
        print(f'Error building supersampling indices for {uid}: {e}')
        kwargs_numerics = {'supersampling_factor': 3, 'compute_mode': 'regular'}

    # load the lens based on uid
    lens = util.unpickle(os.path.join(input_dir, f'lens_with_subhalos_{uid}.pkl'))
    assert lens.uid == uid, f'UID mismatch: {lens.uid} != {uid}'

    # generate lenstronomy model and save
    try:
        for band in bands:
            zp = sca_zp_dict[band]
            if pieces:
                model, lens_surface_brightness, source_surface_brightness = lens.get_array(
                    num_pix=num_pix * grid_oversample, side=side, band=band, zp=zp, return_pieces=True, kwargs_numerics=kwargs_numerics)
                np.save(os.path.join(output_dir, f'array_{lens.uid}_lens_{band}'), lens_surface_brightness)
                np.save(os.path.join(output_dir, f'array_{lens.uid}_source_{band}'), source_surface_brightness)
            else:
                model = lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band, zp=zp, kwargs_numerics=kwargs_numerics)
            np.save(os.path.join(output_dir, f'array_{lens.uid}_{band}'), model)
    except Exception as e:
        print(f'Error generating model for {lens.uid}: {e}')
        return

    # pickle lens to save attributes updated by get_array()
    try:
        pickle_target_lens = os.path.join(output_dir, f'lens_{lens.uid}.pkl')
        util.pickle(pickle_target_lens, lens)
    except Exception as e:
        print(f'Error pickling lens {lens.uid}: {e}')
        return


if __name__ == '__main__':
    main()
