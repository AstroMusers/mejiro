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
    from mejiro.utils import util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    limit = pipeline_params['limit']

    # directories to get pickled lenses (with subhalos) from
    input_parent_dir = config.machine.dir_02
    sca_dirnames = [os.path.basename(d) for d in glob(os.path.join(input_parent_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dirnames])
    scas = [str(sca).zfill(2) for sca in scas]

    # directories to write the output to
    output_parent_dir = config.machine.dir_03
    util.create_directory_if_not_exists(output_parent_dir)
    util.clear_directory(output_parent_dir)
    for sca in scas:
        os.makedirs(os.path.join(output_parent_dir, f'sca{sca}'), exist_ok=True)

    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(config.machine.dir_02 + f'/sca{sca}/lens_with_subhalos_*.pkl'))
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
    # TODO having resource issues here as well
    process_count -= 10
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_dir = os.path.join(input_parent_dir, f'sca{sca}')
        output_dir = os.path.join(output_parent_dir, f'sca{sca}')
        for uid in lens_uids:
            tuple_list.append((uid, pipeline_params, input_dir, output_dir))
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
    for batch in tqdm(batches):  # TODO tqdm thinks that there are all of the lenses, even when a limit is set
        pool = Pool(processes=process_count)
        pool.map(get_model, batch)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_model(input):
    from mejiro.utils import util

    # unpack tuple
    (uid, pipeline_params, input_dir, output_dir) = input

    # unpack pipeline params
    bands = pipeline_params['bands']
    num_pix = pipeline_params['num_pix']
    side = pipeline_params['side']
    grid_oversample = pipeline_params['grid_oversample']
    pieces = pipeline_params['pieces']

    # load the lens based on uid
    lens = util.unpickle(os.path.join(input_dir, f'lens_with_subhalos_{uid}.pkl'))
    assert lens.uid == uid, f'UID mismatch: {lens.uid} != {uid}'

    # generate lenstronomy model and save
    for band in bands:
        if pieces:
            model, lens_surface_brightness, source_surface_brightness = lens.get_array(
                num_pix=num_pix * grid_oversample, side=side, band=band, return_pieces=True)
            np.save(os.path.join(output_dir, f'array_{lens.uid}_lens_{band}'), lens_surface_brightness)
            np.save(os.path.join(output_dir, f'array_{lens.uid}_source_{band}'), source_surface_brightness)
        else:
            model = lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
        np.save(os.path.join(output_dir, f'array_{lens.uid}_{band}'), model)

    # pickle lens to save attributes updated by get_array()
    pickle_target_lens = os.path.join(output_dir, f'lens_{lens.uid}.pkl')
    util.pickle(pickle_target_lens, lens)


if __name__ == '__main__':
    main()
