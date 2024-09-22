import pytest

from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage
from mejiro.instruments.roman import Roman


def test_set_up_pixel_grid():
    strong_lens = SampleStrongLens()
    instrument = Roman()

    # no oversampling
    synthetic_image = SyntheticImage(strong_lens=strong_lens, 
                                     instrument=instrument, 
                                     band='F129', 
                                     arcsec=4.95,
                                     oversample=1)  # 4.95 / 0.11 = 45 so expect 4 pixels, then 4.95 arcsec
    assert synthetic_image.arcsec == 4.95
    assert synthetic_image.num_pix == 45
    
    synthetic_image = SyntheticImage(strong_lens=strong_lens, 
                                     instrument=instrument, 
                                     band='F129', 
                                     arcsec=5.0,
                                     oversample=1)  # 5 / 0.11 = 45.45 so expect 46 pixels, then 47 so odd, then 5.17 arcsec
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.num_pix == 47

    # oversampling
    synthetic_image = SyntheticImage(strong_lens=strong_lens, 
                                    instrument=instrument, 
                                    band='F129', 
                                    arcsec=4.95,
                                    oversample=3)
    assert synthetic_image.arcsec == 4.95
    assert synthetic_image.num_pix == 135

    synthetic_image = SyntheticImage(strong_lens=strong_lens, 
                                    instrument=instrument, 
                                    band='F129', 
                                    arcsec=5.0,
                                    oversample=3)
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.num_pix == 141

    # unhappy paths
    with pytest.raises(AssertionError):
        synthetic_image = SyntheticImage(strong_lens=strong_lens, 
                                         instrument=instrument, 
                                         band='F129', 
                                         arcsec=5.0,
                                         oversample=2)  # even oversampling should raise an error
        
    with pytest.raises(AssertionError):
        synthetic_image = SyntheticImage(strong_lens=strong_lens, 
                                         instrument=instrument, 
                                         band='F129', 
                                         arcsec=5.0,
                                         oversample=0.1)
        