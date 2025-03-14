import datetime
import hydra
import multiprocessing
import numpy as np
import os
import sys
import time
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    limit = pipeline_params['limit']
    rgb_bands = pipeline_params['rgb_bands']
    assert len(rgb_bands) == 3, 'rgb_bands must be a list of 3 bands'

    # set nice level
    os.nice(pipeline_params['nice'])

    # directories to read from
    if debugging:
        input_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '04')
    else:
        input_parent_dir = config.machine.dir_04
    sca_dirnames = [os.path.basename(d) for d in glob(os.path.join(input_parent_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dirnames])
    scas = [str(sca).zfill(2) for sca in scas]

    # directories to write the output to
    if debugging:
        output_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '05')
    else:
        output_parent_dir = config.machine.dir_05
    util.create_directory_if_not_exists(output_parent_dir)
    util.clear_directory(output_parent_dir)
    for sca in scas:
        os.makedirs(os.path.join(output_parent_dir, f'sca{sca}'), exist_ok=True)

    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_parent_dir + f'/sca{sca}/galsim_*.npy'))
        lens_uids = [os.path.basename(i).split('_')[1] for i in pickled_lenses]
        lens_uids = list(set(lens_uids))  # remove duplicates
        lens_uids = sorted(lens_uids)
        uid_dict[sca] = lens_uids

    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)

    if limit != 'None' and limit < count:
        count = limit

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
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
    execution_times = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_image, batch):
            execution_times.append(output)

    # TODO update and append results from each batch, instead of writing all at end; or maybe this is fine for the number of images we'll be generating
    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '05',
                              os.path.join(os.path.dirname(output_parent_dir), 'execution_times.json'))


def get_image(input):
    start = time.time()

    # unpack tuple
    (uid, pipeline_params, input_dir, output_dir) = input
    rgb_bands = pipeline_params['rgb_bands']
    stretch = pipeline_params['rgb_stretch']
    q = pipeline_params['rgb_q']
    pieces = pipeline_params['pieces']

    # assign bands to colors
    red = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{rgb_bands[0]}.npy')
    green = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{rgb_bands[1]}.npy')
    blue = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{rgb_bands[2]}.npy')

    # generate and save color image
    from mejiro.helpers import color
    rgb_image = color.get_rgb(image_b=blue, image_g=green, image_r=red, stretch=stretch,
                              Q=q)  # originally stretch=4, Q=5; then stretch=3, Q=4
    np.save(os.path.join(output_dir, f'galsim_color_{str(uid).zfill(8)}.npy'), rgb_image)

    if pieces:
        for piece in ['lens', 'source']:
            red = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{piece}_{rgb_bands[0]}.npy')
            green = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{piece}_{rgb_bands[1]}.npy')
            blue = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{piece}_{rgb_bands[2]}.npy')
            rgb_image = color.get_rgb(image_b=blue, image_g=green, image_r=red, stretch=stretch, Q=q)
            np.save(os.path.join(output_dir, f'galsim_color_{str(uid).zfill(8)}_{piece}.npy'), rgb_image)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    return execution_time


if __name__ == '__main__':
    main()
