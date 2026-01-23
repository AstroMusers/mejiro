from abc import ABC, abstractmethod


class Instrument(ABC):
    def __init__(
            self,
            name,
            bands,
            num_detectors,
            engines
    ):
        self.name = name
        self.bands = bands
        self.num_detectors = num_detectors
        self.engines = engines
        self.versions = {}

    def convert_mag_to_cps(self, magnitude, band):
        from lenstronomy.Util.data_util import magnitude2cps
        zeropoint = self.get_zeropoint_magnitude(band)
        return magnitude2cps(magnitude, zeropoint)

    def convert_cps_to_mag(self, cps, band):
        from lenstronomy.Util.data_util import cps2magnitude
        zeropoint = self.get_zeropoint_magnitude(band)
        return cps2magnitude(cps, zeropoint)

    @abstractmethod
    def get_pixel_scale(self, band):
        """
        Get the pixel scale for a given band in arcsec/pix.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            Pixel scale in arcsec/pix.
        """
        pass

    @abstractmethod
    def get_psf_kwargs(self, band, **kwargs):
        """
        Get the keyword arguments for generating the Point Spread Function (PSF) for a given band.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        dict
            Keyword arguments for generating the PSF.
        """
        pass

    @abstractmethod
    def get_psf_fwhm(self, band):
        """
        Get the Full Width at Half Maximum (FWHM) of the Point Spread Function (PSF) for a given band in arcsec.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            PSF FWHM in arcsec.
        """
        pass

    @abstractmethod
    def get_thermal_background(self, band):
        """
        Get the thermal background for a given band in ct/pix.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            Thermal background in ct/pix.
        """
        pass

    @abstractmethod
    def get_zeropoint_magnitude(self, band):
        """
        Get the zeropoint magnitude for a given band.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        float
            The zeropoint magnitude for the specified band.
        """
        pass

    @abstractmethod
    def load_speclite_filters():
        """
        Load the filter response curves for the instrument to speclite.

        Returns
        -------
        speclite.filters.FilterSequence
            The loaded filter response curves.
        """
        pass


    @abstractmethod
    def default_params():
        """
        Returns a dictionary of default parameters for the instrument. For example, for the Roman WFI, the 'detector' parameter enables the calculation of detector-specific fluxes from the detector-specific zero-point magnitude.

        Returns
        -------
        dict
            The default parameters.
        """
        pass

    @abstractmethod
    def validate_instrument_params(params):
        """
        Validate the parameters for an instrument. If any required parameters are missing, they will be populated with the default values (see default_params). If any parameters are invalid, a ValueError will be raised.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters to be validated.

        Returns
        -------
        dict
            The validated parameters.
        """
        pass
