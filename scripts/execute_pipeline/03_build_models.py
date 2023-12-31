import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import hydra
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

    # directory to write the output to
    output_dir = config.machine.dir_03
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # open pickled lens list
    lens_list = util.unpickle_all(config.machine.dir_02)

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for i, _ in enumerate(lens_list):
        tuple_list.append((lens_list[i], pipeline_params, output_dir))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(get_model, batch)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_model(input):
    from mejiro.utils import util

    # unpack tuple
    (lens, pipeline_params, output_dir) = input

    # unpack pipeline params
    num_pix = pipeline_params['num_pix']
    side = pipeline_params['side']
    grid_oversample = pipeline_params['grid_oversample']

    # generate lenstronomy model
    model = lens.get_array(num_pix=num_pix * grid_oversample, side=side)

    # pickle the results
    lens_dict = {
        'lens': lens,
        'model': model
    }
    pickle_target = os.path.join(output_dir, f'lens_dict_{lens.uid}_{lens.band}')
    util.pickle(pickle_target, lens_dict)


if __name__ == '__main__':
    main()
