from abc import ABC, abstractmethod


class Instrument(ABC):
    """Abstract base class for all instrument implementations.

    Defines the common interface shared by all supported telescopes and
    instruments (Roman, HST, JWST, LSST, HWO). Concrete subclasses must
    implement the abstract methods for band-specific optical and detector
    properties.
    """

    def __init__(
            self,
            name,
            bands,
            num_detectors,
            engines
    ):
        """Initialize an Instrument.

        Parameters
        ----------
        name : str
            Human-readable name of the instrument (e.g. ``'Roman WFI'``).
        bands : list of str
            Filter band identifiers supported by the instrument
            (e.g. ``['F062', 'F087', 'F106']`` for Roman).
        num_detectors : int
            Number of detector chips in the focal plane array.
        engines : list
            Simulation engine instances or identifiers that can be used
            with this instrument.
        """
        self.name = name
        self.bands = bands
        self.num_detectors = num_detectors
        self.engines = engines
        self.versions = {}

    def convert_mag_to_cps(self, magnitude, band):
        """Convert an AB magnitude to counts per second for a given band.

        Parameters
        ----------
        magnitude : float
            Source brightness in AB magnitudes.
        band : str
            Filter band identifier (e.g. ``'F129'`` for Roman).

        Returns
        -------
        float
            Source flux in counts per second (ct/s).
        """
        from lenstronomy.Util.data_util import magnitude2cps
        zeropoint = self.get_zeropoint_magnitude(band)
        return magnitude2cps(magnitude, zeropoint)

    def convert_cps_to_mag(self, cps, band):
        """Convert counts per second to an AB magnitude for a given band.

        Parameters
        ----------
        cps : float
            Source flux in counts per second (ct/s).
        band : str
            Filter band identifier (e.g. ``'F129'`` for Roman).

        Returns
        -------
        float
            Source brightness in AB magnitudes.
        """
        from lenstronomy.Util.data_util import cps2magnitude
        zeropoint = self.get_zeropoint_magnitude(band)
        return cps2magnitude(cps, zeropoint)

    def get_stray_light_fraction(self):
        """Get the stray light fraction for the instrument.

        Returns
        -------
        float
            Fraction of stray light. Subclasses are expected to set
            ``self.stray_light_fraction`` in their ``__init__``.
        """
        return self.stray_light_fraction

    @abstractmethod
    def get_pixel_scale(self, band):
        """
        Get the pixel scale for a given band in arcsec/pix.

        Subclasses with a single pixel scale across all bands may ignore
        the ``band`` argument.

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
    def get_dark_current(self, band):
        """
        Get the dark current for a given band.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            Dark current per pixel per second.
        """
        pass

    @abstractmethod
    def get_read_noise(self, band):
        """
        Get the read noise for a given band.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            Read noise per pixel.
        """
        pass

    @abstractmethod
    def get_gain(self, band):
        """
        Get the detector gain for a given band.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        float
            Detector gain.
        """
        pass

    @abstractmethod
    def get_sky_level(self, band):
        """
        Get the sky background level for a given band.

        Parameters
        ----------
        band : str
            The name of the band, e.g., 'F129' for Roman or 'J' for HWO.

        Returns
        -------
        Astropy.units.Quantity
            Sky background level in ct/pix.
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

    @staticmethod
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


    @staticmethod
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

    @staticmethod
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
