import os
import sys
import time

import hydra
from tqdm import tqdm
from glob import glob


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir

    # check if mejiro in repo_dir
    try:
        if repo_dir not in sys.path:
            sys.path.append(repo_dir)
        import mejiro
        module_path = os.path.dirname(mejiro.__file__)
        print(f'Found mejiro at {module_path}')
    except:
        raise Exception(f'Could not find mejiro in {repo_dir}')

    # check Pandeia installation
    from pandeia import engine
    print(engine.pandeia_version())
    # TODO fix
    # if 'ENVIRONMENT VARIABLE UNSET' in engine.pandeia_version():
    #     raise Exception('Environment variable unset')


if __name__ == '__main__':
    main()
