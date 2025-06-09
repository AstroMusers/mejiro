import hydra
import numpy as np
import os
import sys
import time
from glob import glob
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir
    data_dir = config.machine.data_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util
    from mejiro.lenses import lens_util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    # set nice level
    os.nice(pipeline_params['nice'])

    # create directories
    export_dir = os.path.join(data_dir, 'h5_export')
    util.create_directory_if_not_exists(export_dir)
    subhalo_dir = os.path.join(export_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.clear_directory(subhalo_dir)

    # get all lenses
    all_lenses = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=True, verbose=True, limit=None,
                                                 exposure=True)

    # filter lenses by SNR
    all_lenses = [lens for lens in all_lenses if lens.snr != np.inf]

    print(f'Copying pickled subhalo realizations for {len(all_lenses)} systems')
    for lens in tqdm(all_lenses):
        uid = lens.uid

        subhalo_files = glob(
            os.path.join(pipeline_dir, '02', '*', 'subhalos', f'subhalo_realization_{str(uid).zfill(8)}.pkl'))
        subhalo_filepath = subhalo_files[0]

        dest_path = os.path.join(subhalo_dir, os.path.basename(subhalo_filepath))
        os.system(f'cp {subhalo_filepath} {dest_path}')

    # for now, just tar the subhalo directory via the command line: `tar -czvf subhalos_v_0_0_3.tar.gz subhalos/`

    # filepath = os.path.join(export_dir, 'subhalos_v_0_0_3.tar.gz')
    # if os.path.exists(filepath):
    #     os.remove(filepath)

    # print(f'Creating tarball {filepath}...')
    # with tarfile.open(filepath, "w:gz") as tar:
    #     tar.add(subhalo_dir)
    # print(f'Created tarball {filepath}')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
