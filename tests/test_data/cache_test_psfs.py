import hydra
import sys


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.engines import webbpsf_engine

    bands = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']

    for band in bands:
        psf_id = webbpsf_engine.get_psf_id(band, 1, (2048, 2048), 5, 101)
        webbpsf_engine.cache_psf(psf_id, '')


if __name__ == '__main__':
    main()
