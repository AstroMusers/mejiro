import os

import mejiro
from mejiro.exposure import Exposure
from mejiro.galaxy_galaxy import SampleGG
from mejiro.instruments.jwst import JWST
from mejiro.synthetic_image import SyntheticImage

jwst = JWST()

# Use one of the JWST PSFs that ship with mejiro (oversample 1, 41 x 41 pixels), so no
# STPSF data files are needed.
psf_cache_dir = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'psfs', 'jwst')
kwargs_psf = jwst.get_psf_kwargs('F200W', oversample=1, num_pix=41,
                                 check_cache=True, psf_cache_dir=psf_cache_dir)

synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                 instrument=jwst,
                                 band='F200W',
                                 fov_arcsec=5,
                                 kwargs_psf=kwargs_psf)

exposure = Exposure(synthetic_image,
                    exposure_time=1000,
                    engine='galsim')
