from abc import ABC, abstractmethod


class Instrument(ABC):
    def __init__(
            self,
            name,
            bands,
            engines
    ):
        self.name = name
        self.bands = bands
        self.engines = engines
        self.versions = {}
    
    @abstractmethod
    def get_minimum_zodiacal_light(self, band):
        """
        Get the minimum zodiacal light at high galactic latitudes for a given band in ct/pix.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            Minimum zodiacal light in ct/pix.
        """
        pass

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
        Get the zeropoint AB magnitude for a given band.

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
    def default_params():
        pass

    @abstractmethod
    def validate_instrument_params(params):
        pass
