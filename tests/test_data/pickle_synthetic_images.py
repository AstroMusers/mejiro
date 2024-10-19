import sys

import hydra


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.instruments.roman import Roman
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils import util

    roman = Roman()
    lens = SampleStrongLens()

    synthetic_image = SyntheticImage(strong_lens=lens,
                                        instrument=roman,
                                        band='F129',
                                        arcsec=5,
                                        oversample=1,
                                        verbose=True)
    
    util.pickle('synthetic_image_roman_F129_5_1.pkl', synthetic_image)

    synthetic_image = SyntheticImage(strong_lens=lens,
                                        instrument=roman,
                                        band='F129',
                                        arcsec=5,
                                        oversample=5,
                                        verbose=True)
    
    util.pickle('synthetic_image_roman_F129_5_5.pkl', synthetic_image)


if __name__ == '__main__':
    main()
