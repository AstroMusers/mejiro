from mejiro.galaxy_galaxy import SampleGG
from mejiro.instruments.roman import Roman
from mejiro.synthetic_image import SyntheticImage
from mejiro.exposure import Exposure

synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                 instrument=Roman(),
                                 band='F129',
                                 fov_arcsec=5)

exposure = Exposure(synthetic_image,
                    exposure_time=146,
                    engine='galsim')
