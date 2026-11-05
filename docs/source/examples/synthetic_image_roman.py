from mejiro.galaxy_galaxy import SampleGG
from mejiro.instruments.roman import Roman
from mejiro.synthetic_image import SyntheticImage

strong_lens = SampleGG()

synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                 instrument=Roman(),
                                 band='F129',
                                 fov_arcsec=5)
synthetic_image.plot()
