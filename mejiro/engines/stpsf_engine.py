import numpy as np
import os
import warnings
from glob import glob
from stpsf.roman import WFI

from mejiro.engines.engine import Engine
from mejiro.utils import roman_util, lenstronomy_util


class STPSFEngine(Engine):
    @staticmethod
    def defaults(instrument_name):
        if instrument_name.lower() == 'roman':
            return {}  # TODO implement
        else:
            Engine.instrument_not_supported(instrument_name)
    
    @staticmethod
    def validate_engine_params(engine_params):
        # TODO implement
        pass

    @staticmethod
    def get_roman_psf_kwargs(band, detector, detector_position, oversample, num_pix, check_cache=False, psf_cache_dir=None,
                    verbose=False):
        kernel = STPSFEngine.get_roman_psf(band, detector, detector_position, oversample, num_pix,
                                            check_cache=check_cache, psf_cache_dir=psf_cache_dir, verbose=verbose)
        return lenstronomy_util.get_pixel_psf_kwargs(kernel, oversample)

    @staticmethod
    def get_roman_psf(band, detector, detector_position, oversample, num_pix, check_cache=False, psf_cache_dir=None,
                    verbose=False, **calc_psf_kwargs):
        """
        Generate a Roman WFI PSF using WebbPSF.

        Parameters
        ----------
        band : str
            The band.
        detector : int
            The detector number.
        detector_position : tuple of int
            The (x, y) position on the detector.
        oversample : int
            The oversampling factor.
        num_pix : int
            The number of pixels on a side. This parameter is passed to WebbPSF's `fov_pixels` parameter.
        check_cache : bool, optional
            If True, check the cache for an existing PSF before generating a new one. Default is True.
        psf_cache_dir : str, optional
            The directory where cached PSFs are stored. If None, defaults to the directory installed with mejiro. Default is None.
        verbose : bool, optional
            If True, print additional information. Default is False.
        **calc_psf_kwargs : dict
            Additional keyword arguments to pass to WebbPSF's `calc_psf` method.

        Returns
        -------
        np.ndarray
            The PSF kernel.
        """
        # first, check if it exists in the cache
        if check_cache:
            assert psf_cache_dir is not None, 'Must provide a PSF cache directory if checking the cache'
            psf_id = STPSFEngine.get_psf_id(band, detector, detector_position, oversample, num_pix)
            cached_psf = STPSFEngine.get_cached_psf(psf_id, psf_cache_dir, verbose)
            if cached_psf is not None:
                return cached_psf

        # set PSF parameters
        wfi = WFI()
        wfi.filter = band.upper()
        wfi.detector = roman_util.get_sca_string(detector)
        wfi.detector_position = detector_position
        wfi.options['output_mode'] = 'oversampled'

        # generate PSF in WebbPSF
        psf = wfi.calc_psf(fov_pixels=num_pix, oversample=oversample, **calc_psf_kwargs)

        return psf['OVERSAMP'].data


    @staticmethod
    def get_psf_id(band, detector, detector_position, oversample, num_pix):
        """
        Generate a PSF identifier string. mejiro's Roman simulation uses this under-the-hood to cache and retrieve Roman PSFs.

        Parameters
        ----------
        band : str
            The band.
        detector : str
            The detector number.
        detector_position : tuple of int
            The (x, y) position on the detector.
        oversample : int
            The oversampling factor.
        num_pix : int
            The number of pixels on a side. 

        Returns
        -------
        str
            A unique identifier string for the PSF.
        """
        detector = roman_util.get_sca_int(detector)
        return f'{band}_{detector}_{detector_position[0]}_{detector_position[1]}_{oversample}_{num_pix}'


    @staticmethod
    def get_params_from_psf_id(psf_id):
        """
        Converts mejiro's Roman PSF identifier string format back to a list of PSF parameters.

        Parameters
        ----------
        psf_id : str
            mejiro's Roman PSF identifier string.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - band (str): The band.
            - detector (int): The detector number.
            - detector_position (tuple of int): The (x, y) position on the detector.
            - oversample (int): The oversampling factor.
            - num_pix (int): The number of pixels on a side.
        """
        band, detector, detector_position_0, detector_position_1, oversample, num_pix = psf_id.split('_')
        return band, int(detector), (int(detector_position_0), int(detector_position_1)), int(oversample), int(num_pix)


    @staticmethod
    def get_roman_psf_from_id(psf_id, check_cache=True, psf_cache_dir=None, verbose=False, **calc_psf_kwargs):
        """
        Wrapper method for `get_roman_psf` that accepts the PSF's identifier string.

        Parameters
        ----------
        psf_id : str
            The identifier for the PSF, which encodes various parameters.
        check_cache : bool, optional
            If True, check the cache for an existing PSF before generating a new one. Default is True.
        psf_cache_dir : str, optional
            The directory where cached PSFs are stored. If None, defaults to the directory installed with mejiro. Default is None.
        verbose : bool, optional
            If True, print additional information. Default is False.
        **calc_psf_kwargs : dict
            Additional keyword arguments to pass to WebbPSF's `calc_psf` method.

        Returns
        -------
        np.ndarray
            The PSF kernel.
        """
        band, detector, detector_position, oversample, num_pix = STPSFEngine.get_params_from_psf_id(psf_id)
        return STPSFEngine.get_roman_psf(band, detector, detector_position, oversample, num_pix, check_cache, psf_cache_dir, verbose,
                            **calc_psf_kwargs)


    @staticmethod
    def cache_psf(id_string, psf_cache_dir, verbose=True):
        """
        Save a PSF to the provided directory.

        Parameters
        ----------
        id_string : str
            The PSF identifier string.
        psf_cache_dir : str
            The directory where cached PSFs are stored.
        verbose : bool, optional
            If True, print messages about the caching process. Default is True.

        Returns
        -------
        None
        """
        psf_path = os.path.join(psf_cache_dir, f'{id_string}.npy')
        if os.path.exists(psf_path):
            if verbose:
                print(f'PSF {id_string} already cached to {psf_path}')
        else:
            psf = STPSFEngine.get_roman_psf_from_id(id_string, check_cache=False, verbose=verbose)
            np.save(psf_path, psf)
            if verbose:
                print(f'Cached PSF to {psf_path}')


    @staticmethod
    def get_cached_psf(id_string, psf_cache_dir, verbose):
        """
        Check if a PSF exists in the provided cache directory. If found, load and return it. Otherwise, return None.

        Parameters
        ----------
        id_string : str
            The PSF identifier string.
        psf_cache_dir : str or None
            The directory where cached PSFs are stored. If None, defaults to the directory installed with mejiro.
        verbose : bool
            If True, print additional information.

        Returns
        -------
        numpy.ndarray or None
            The cached PSF if found, otherwise None.
        """
        # if no psf cache directory provided, default to those installed with mejiro
        if psf_cache_dir is None:
            import mejiro
            module_path = os.path.dirname(mejiro.__file__)
            psf_cache_dir = os.path.join(module_path, 'data', 'cached_psfs')

        psf_path = glob(os.path.join(psf_cache_dir, f'{id_string}.npy'))
        if len(psf_path) == 1:
            if verbose:
                print(f'Loading cached PSF: {psf_path[0]}')
            return np.load(psf_path[0])
        else:
            band, detector, detector_position, oversample, num_pix = STPSFEngine.get_params_from_psf_id(id_string)
            warnings.warn(
                f'PSF {band} SCA{str(detector).zfill(2)} {detector_position} {oversample} {num_pix} not found in cache {psf_cache_dir}')  # TODO change to logging
            if verbose:
                print(
                    f'PSF {band} SCA{str(detector).zfill(2)} {detector_position} {oversample} {num_pix} not found in cache {psf_cache_dir}')
            return None
