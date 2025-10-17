import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from mejiro.analysis import regions
from mejiro.cosmo import cosmo


class StrongLens(ABC):
    """
    Parent class for strong lenses.

    At minimum, a unique name and the parameterization in lenstronomy (`kwargs_model` and `kwargs_params`) must be provided. If the light models have amplitudes (`amp`), they will be used. If not, magnitudes with their corresponding filters must be provided in the 'physical_params' dictionary. If both are provided, the `amp` values will be used.

    Parameters
    ----------
    name : str
        The name of the galaxy-galaxy strong lens. Should be unique.
    coords : astropy.coordinates.SkyCoord or None
        The coordinates of the system.
    kwargs_model : dict
        In Lenstronomy format: see Lenstronomy documentation.
    kwargs_params : dict
        In Lenstronomy format: see Lenstronomy documentation.
    physical_params : dict
        A dictionary of physical parameters. The following are populated when importing from SLSim:
            - 'lens_stellar_mass': the stellar mass of the lens galaxy in solar masses.
            - 'lens_velocity_dispersion': the velocity dispersion of the lens galaxy in km/s.
            - 'magnification': the magnification of the source galaxy.
            - 'magnitudes': a dictionary of magnitudes for the lens and source galaxies, with keys 'lens' and 'source', respectively. Each value should be a dictionary with keys corresponding to the filter names (e.g., 'F062', 'F087', etc.) and values as the magnitudes in those filters.
    use_jax : bool, or list of bool
        Whether to use JAXtronomy for calculations. Default is None, then set to True in the constructor for all lens model element(s). See the `lenstronomy documentation <https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LensModel.html#module-lenstronomy.LensModel.lens_model>`__ for details.

    Notes
    -----

    - Note that JAXtronomy is utilized by default for all lens model elements. To disable this, either set `use_jax` to False (which will disable it for all lens model elements) or provide a list of booleans specifying which lens model(s) to disable JAXtronomy for.
        
    Examples
    --------
    Here is a sample ``physical_params`` dictionary:

    .. code-block:: python

        {
            "lens_stellar_mass": 977409654003.7206,
            "lens_velocity_dispersion": 318.1035069180407,
            "magnification": 2.1572878148080696,
            "magnitudes": {
                "lens": {
                    "F062": 25.31638290929369,
                    "F087": 24.96369787259651,
                    "F106": 23.9596060901365,
                    "F129": 22.97223751280789,
                    "F146": 22.29807037949136,
                    "F158": 21.906108678706293,
                    "F184": 21.45793326374795,
                    "F213": 21.118396579379453
                },
                "lensed_source": {
                    "F062": 25.500925653675434,
                    "F087": 25.177803863752867,
                    "F106": 25.00159434812006,
                    "F129": 24.8952958657035,
                    "F146": 24.766725445488536,
                    "F158": 24.667238282726974,
                    "F184": 24.465433844571177,
                    "F213": 24.029905125404884
                },
                "source": {
                    "F062": 26.33569587971921,
                    "F087": 26.012574089796644,
                    "F106": 25.836364574163838,
                    "F129": 25.730066091747275,
                    "F146": 25.601495671532312,
                    "F158": 25.50200850877075,
                    "F184": 25.300204070614953,
                    "F213": 24.86467535144866
                }
            }
        }

    """
    def __init__(
            self,
            name,
            coords,
            kwargs_model,
            kwargs_params,
            physical_params,
            use_jax
    ):
        self.name = name
        self.coords = coords
        self.kwargs_model = kwargs_model
        self.kwargs_params = kwargs_params
        self.physical_params = physical_params
        self.use_jax = use_jax

        # set cosmology
        if 'cosmo' in kwargs_model:
            self.cosmo = kwargs_model['cosmo']
        else:
            raise ValueError("Set astropy.cosmology instance in kwargs_model['cosmo']")
        
        # set JAXtronomy flag
        if self.use_jax is None:
            self.use_jax = [True] * len(self.lens_model_list)  # default to True for all lens models
        elif isinstance(self.use_jax, list):
            if len(self.use_jax) != len(self.lens_model_list):
                raise ValueError("Length of use_jax list must match the number of lens models.")
        elif isinstance(self.use_jax, bool):
            self.use_jax = [self.use_jax] * len(self.lens_model_list)
        else:
            raise ValueError("use_jax must be a boolean or a list of booleans.")
        
        # check that brightnesses are provided somewhere
        _ = self.validate_light_models()

        # fields to initialize: these can be computed on demand
        self.lens_cosmo = None
        self.realization = None
        self.kwargs_lens_macromodel = None
        self.lens_redshift_list_macromodel = None
        self.lens_model_list_macromodel = None
        self.lens_model_macromodel = None

    def get_f_sub(self, fov_arcsec=5, num_pix=100, plot=False):
        if self.realization is None:
            raise ValueError("No realization has been added. Use `add_realization()` to add a pyHalo realization.")

        einstein_radius = self.get_einstein_radius()
        r_in = (einstein_radius - 0.2) / (fov_arcsec / num_pix)  # units of pixels
        r_out = (einstein_radius + 0.2) / (fov_arcsec / num_pix)  # units of pixels

        halo_lens_model_list, _, _, _ = self.realization.lensing_quantities(add_mass_sheet_correction=False)
        macrolens_lens_model_list = self.lens_model_list[:-len(halo_lens_model_list)]
        lens_model_macro = LensModel(lens_model_list=macrolens_lens_model_list,
                                        z_lens=self.z_lens,
                                        z_source=self.z_source,
                                        lens_redshift_list=self.z_lens * len(macrolens_lens_model_list),
                                        cosmo=self.cosmo,
                                        multi_plane=False)
        _r = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, num_pix)
        xx, yy = np.meshgrid(_r, _r)
        macrolens_kappa = lens_model_macro.kappa(xx.ravel(), yy.ravel(), self.kwargs_lens[:-len(halo_lens_model_list)]).reshape(num_pix, num_pix)
        subhalo_kappa = self.get_realization_kappa(fov_arcsec=5, num_pix=100, add_mass_sheet_correction=False)

        mask = regions.annular_mask(num_pix, num_pix, (num_pix // 2, num_pix // 2), r_in, r_out)
        masked_kappa_subhalos = np.ma.masked_array(subhalo_kappa, mask=~mask)
        masked_kappa_macro = np.ma.masked_array(macrolens_kappa, mask=~mask)
        f_sub = masked_kappa_subhalos.compressed().sum() / masked_kappa_macro.compressed().sum()

        if plot:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
            ax[0].imshow(masked_kappa_subhalos, cmap='bwr')
            ax[1].imshow(masked_kappa_macro, cmap='bwr')
            ax[0].set_title(
                r'$\sum_n \int_{\mathrm{annulus}}d^2\theta\,\kappa_n=$' + f'{masked_kappa_subhalos.compressed().sum():.6f}')
            ax[1].set_title(
                r'$\int_{\mathrm{annulus}}d^2\theta\,\kappa_{\mathrm{host}}=$' + f'{masked_kappa_macro.compressed().sum():.6f}')
            for a in ax: a.axis('off')
            plt.suptitle(r'$f_{\mathrm{sub}}=$' + f'{f_sub:.6f}')
            return f_sub, ax
        else:
            return f_sub, None


    def get_kappa(self, fov_arcsec=5, num_pix=100):
        """
        Computes the convergence (kappa) map of the lens model over a specified field of view.

        Parameters
        ----------
        fov_arcsec : float, optional
            The field of view in arcseconds. Default is 5.
        num_pix : int, optional
            The number of pixels along each axis for the output grid. Default is 100.

        Returns
        -------
        kappa_map : ndarray
            A 2D array of shape (num_pix, num_pix) representing the convergence (kappa) values
            computed over the grid.

        Notes
        -----
        The method creates a square grid centered at (0, 0) in arcseconds, evaluates the lens model's
        convergence at each grid point, and returns the resulting map.
        """
        _r = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, num_pix)
        xx, yy = np.meshgrid(_r, _r)
        return self.lens_model.kappa(xx.ravel(), yy.ravel(), self.kwargs_lens).reshape(num_pix, num_pix)

    def get_realization_kappa(self, fov_arcsec=5, num_pix=100, add_mass_sheet_correction=False):
        """
        Computes the convergence (kappa) map for the current realization over a specified field of view.

        Parameters
        ----------
        fov_arcsec : float, optional
            The field of view in arcseconds for the kappa map. Default is 5.
        num_pix : int, optional
            The number of pixels per side for the output kappa map. Default is 100.
        add_mass_sheet_correction : bool, optional
            Whether to include the mass sheet correction in the realization. Default is False.

        Returns
        -------
        kappa_map : numpy.ndarray
            A 2D array of shape (num_pix, num_pix) representing the convergence (kappa) map.

        Raises
        ------
        ValueError
            If no realization has been added prior to calling this method.

        Notes
        -----
        This method requires a pyHalo realization to be added using `add_realization()`.
        """
        if self.realization is None:
            raise ValueError("No realization has been added. Use `add_realization()` to add a pyHalo realization.")

        halo_lens_model_list, halo_redshift_array, kwargs_halos, _ = self.realization.lensing_quantities(add_mass_sheet_correction=add_mass_sheet_correction)
        halo_redshift_list = list(halo_redshift_array)

        lens_model_realization = LensModel(lens_model_list=halo_lens_model_list,
                                          z_lens=self.z_lens,
                                          z_source=self.z_source,
                                          lens_redshift_list=halo_redshift_list,
                                          cosmo=self.cosmo,
                                          multi_plane=True)

        _r = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, num_pix)
        xx, yy = np.meshgrid(_r, _r)
        return lens_model_realization.kappa(xx.ravel(), yy.ravel(), kwargs_halos).reshape(num_pix, num_pix)

    def add_realization(self, realization, add_mass_sheet_correction=True, use_jax=True):
        """
        Add a pyHalo dark matter subhalo realization to the mass model of the system. See the `pyHalo documentation <https://github.com/dangilman/pyHalo>`__ for details.

        Parameters
        ----------
        realization : pyHalo realization object
            See the pyHalo documentation for details.
        add_mass_sheet_correction : bool, optional
            See the pyHalo documentation for details. Default is True.
        use_jax : bool, optional
            Whether to use JAXtronomy for calculations. Default is True.
        """
        self.realization = realization

        # before adding the realization, save the macromodel parameters
        self.kwargs_lens_macromodel = deepcopy(self.kwargs_lens)
        self.lens_redshift_list_macromodel = deepcopy(self.lens_redshift_list)
        self.lens_model_list_macromodel = deepcopy(self.lens_model_list)
        self.lens_model_macromodel = deepcopy(self.lens_model)

        # get lenstronomy lensing quantities
        halo_lens_model_list, halo_redshift_array, kwargs_halos, _ = realization.lensing_quantities(
            add_mass_sheet_correction=add_mass_sheet_correction)
        
        # halo_lens_model_list and kwargs_halos are lists, but halo_redshift_array is ndarray
        halo_redshift_list = list(halo_redshift_array)

        # add subhalos to lenstronomy objects that model the strong lens
        self.kwargs_lens += kwargs_halos
        self.lens_redshift_list += halo_redshift_list
        self.lens_model_list += halo_lens_model_list

        # use JAXtronomy for supported lens models
        if isinstance(use_jax, bool):
            if use_jax:
                from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS
                
                for halo_lens_model in halo_lens_model_list:
                    if halo_lens_model in _JAXXED_MODELS:
                        self.use_jax.append(True)
                    else:
                        self.use_jax.append(False)
            else:
                self.use_jax += [False] * len(halo_lens_model_list)
        else:
            raise ValueError("use_jax must be a boolean.")
        
    def get_lens_magnitude(self, band):
        """
        Returns
        -------
        float
            Magnitude of the lens in the specified photometric band.
        """
        return self.get_magnitude('lens', band)
    
    def get_source_magnitude(self, band):
        """
        Returns the magnitude of the source in the specified photometric band.

        Parameters
        ----------
        band : str
            The name of the photometric band for which to retrieve the source magnitude.

        Returns
        -------
        float
            The magnitude of the source in the specified band.
        """
        return self.get_magnitude('source', band)
    
    def get_lensed_source_magnitude(self, band):
        """
        Returns the magnitude of the lensed source in the specified photometric band.

        Parameters
        ----------
        band : str
            The name of the photometric band for which to retrieve the lensed source magnitude.

        Returns
        -------
        float
            The magnitude of the lensed source in the specified band.
        """
        return self.get_magnitude('lensed_source', band)

    def get_magnitude(self, kind, band):
        """
        Retrieve the magnitude value for a specified kind and photometric band from the physical parameters dictionary.

        Parameters
        ----------
        kind : str
            Options are 'lens', 'source', 'lensed_source'.
        band : str
            e.g., 'F129' (Roman), 'J' (HWO), etc.

        Returns
        -------
        float
            Magnitude

        Raises
        ------
        ValueError
            If magnitudes are not provided in the `physical_params` dictionary.
            If the specified kind is not present in the magnitudes.
            If the specified band is not present for the given kind.
        """
        if self.physical_params.get('magnitudes') is None:
            raise ValueError("Magnitudes are not provided in the `physical_params` dictionary.")
        if self.physical_params['magnitudes'].get(kind) is None:
            raise ValueError(f"{kind} magnitudes are not provided in the `physical_params` dictionary.")
        if self.physical_params['magnitudes'][kind].get(band) is None:
            raise ValueError(f"{kind} magnitudes for band {band} are not provided in the `physical_params` dictionary.")
        return self.physical_params['magnitudes'][kind][band]

    def get_einstein_radius(self):
        """
        Returns the Einstein radius.

        Returns
        -------
        float
            The Einstein radius (``theta_E``) in angular units (often, arcseconds).

        Raises
        ------
        ValueError
            This method currently does not calculate the Einstein radius. Rather, it retrieves it from the attributes. If it has not been stored in these attributes, a ValueError will be raised.
        """
        einstein_radius = self.physical_params.get('einstein_radius', None)
        if einstein_radius is None:
            einstein_radius = self.kwargs_lens[0].get('theta_E', None)
        if einstein_radius is None:
            raise ValueError("Could not find `einstein_radius` in `physical_params` or `theta_E` in `kwargs_lens`")
        return einstein_radius

    def get_velocity_dispersion(self):
        """
        Get the velocity dispersion of the lensing galaxy in km/s.

        Returns
        -------
        float
            The velocity dispersion of the lensing galaxy in km/s.

        Raises
        ------
        ValueError
            If 'lens_velocity_dispersion' is not present in `self.physical_params`.
        """
        if 'lens_velocity_dispersion' not in self.physical_params:
            raise ValueError("Velocity dispersion not found in physical_params. Please provide 'lens_velocity_dispersion' in physical_params.")
        return self.physical_params['lens_velocity_dispersion']

    def get_stellar_mass(self):
        """
        Get the stellar mass of the lensing galaxy in solar masses.

        Returns
        -------
        float
            The stellar mass of the lensing galaxy in solar masses.

        Raises
        ------
        ValueError
            If 'lens_stellar_mass' is not present in `self.physical_params`.
        """
        if 'lens_stellar_mass' not in self.physical_params:
            raise ValueError("Stellar mass not found in physical_params. Please provide 'lens_stellar_mass' in physical_params.")
        return self.physical_params['lens_stellar_mass']

    def get_main_halo_mass(self):
        """
        Returns the main halo mass of the lensing galaxy in solar masses.

        This method first attempts to retrieve the main halo mass from the ``physical_params`` dictionary
        using the ``main_halo_mass`` key. If this value is not present, it will attempt to estimate the
        main halo mass using the stellar mass (``lens_stellar_mass``) and the lens redshift (``z_lens``)
        via the ``cosmo.stellar_to_main_halo_mass`` method, if available.

        Returns
        -------
        float
            The mass of the main halo in solar masses.

        Raises
        ------
        ValueError
            If neither ``main_halo_mass`` nor ``lens_stellar_mass`` are present in ``physical_params``.
        """
        main_halo_mass = self.physical_params.get('main_halo_mass', None)
        if main_halo_mass is None:
            lens_stellar_mass = self.physical_params.get('lens_stellar_mass', None)
            if lens_stellar_mass is not None:
                main_halo_mass = cosmo.stellar_to_main_halo_mass(stellar_mass=lens_stellar_mass, z=self.z_lens, sample=True)
            else:
                raise ValueError("Could not find `main_halo_mass` or `lens_stellar_mass` in physical_params")
        return main_halo_mass
    
    def get_lens_cosmo(self):
        """
        Get or create the LensCosmo instance, a lenstronomy class that supports physical unit calculations.

        Returns
        -------
        LensCosmo
            The lenstronomy.Cosmo.lens_cosmo.LensCosmo instance for the system.
        """
        if self.lens_cosmo is None:
            self.lens_cosmo = LensCosmo(self.z_lens, self.z_source, cosmo=self.cosmo)
        return self.lens_cosmo
    
    def validate_light_models(self):
        """
        Validates the presence of the lenstronomy amplitude ('amp') parameter in all lens and source light model keyword arguments. If magnitude information is not present in ``self.physical_params``, or if it is incomplete (missing 'lens' or 'source' magnitudes), the method ensures that each light model dictionary in ``self.kwargs_lens_light`` and ``self.kwargs_source`` contains an 'amp' key. If any light model is missing the 'amp' parameter, a ValueError is raised.

        Returns
        -------
        amps_provided : bool
            True if all required amplitude parameters are provided. False if magnitudes are provided for both the lens and source.

        Raises
        ------
        ValueError
            If any light model dictionary is missing the 'amp' parameter.
        """
        amps_provided = False

        if ('magnitudes' not in self.physical_params) or ('magnitudes' in self.physical_params and ('lens' not in self.physical_params['magnitudes'] or 'source' not in self.physical_params['magnitudes'])):
            for light_kwargs in self.kwargs_lens_light + self.kwargs_source:
                if 'amp' not in light_kwargs:
                    raise ValueError(f"Missing 'amp' in {light_kwargs}")
            
            # if loop completes without raising an error, then amps are provided
            amps_provided = True
        
        return amps_provided

    @property
    def kwargs_lens(self):
        return self.kwargs_params.get('kwargs_lens', None)
    
    @kwargs_lens.setter
    def kwargs_lens(self, value):
        self.kwargs_params['kwargs_lens'] = value
    
    @property
    def kwargs_lens_light(self):
        return self.kwargs_params.get('kwargs_lens_light', None)
    
    @kwargs_lens_light.setter
    def kwargs_lens_light(self, value):
        self.kwargs_params['kwargs_lens_light'] = value
    
    @property
    def kwargs_source(self):
        return self.kwargs_params.get('kwargs_source', None)
    
    @kwargs_source.setter
    def kwargs_source(self, value):
        self.kwargs_params['kwargs_source'] = value
    
    @property
    def kwargs_ps(self):
        return self.kwargs_params.get('kwargs_ps', None)
    
    @kwargs_ps.setter
    def kwargs_ps(self, value):
        self.kwargs_params['kwargs_ps'] = value
    
    @property
    def kwargs_extinction(self):
        return self.kwargs_params.get('kwargs_extinction', None)
    
    @kwargs_extinction.setter
    def kwargs_extinction(self, value):
        self.kwargs_params['kwargs_extinction'] = value
    
    @property
    def kwargs_special(self):
        return self.kwargs_params.get('kwargs_special', None)
    
    @kwargs_special.setter
    def kwargs_special(self, value):
        self.kwargs_params['kwargs_special'] = value
    
    @property
    def lens_model_list(self):
        return self.kwargs_model.get('lens_model_list', None)
    
    @lens_model_list.setter
    def lens_model_list(self, value):
        self.kwargs_model['lens_model_list'] = value
    
    @property
    def lens_light_model_list(self):
        return self.kwargs_model.get('lens_light_model_list', None)
    
    @lens_light_model_list.setter
    def lens_light_model_list(self, value):
        self.kwargs_model['lens_light_model_list'] = value
    
    @property
    def source_light_model_list(self):
        return self.kwargs_model.get('source_light_model_list', None)
    
    @source_light_model_list.setter
    def source_light_model_list(self, value):
        self.kwargs_model['source_light_model_list'] = value
    
    @property
    def lens_redshift_list(self):
        return self.kwargs_model.get('lens_redshift_list', None)
    
    @lens_redshift_list.setter
    def lens_redshift_list(self, value):
        self.kwargs_model['lens_redshift_list'] = value

    @property
    def source_redshift_list(self):
        return self.kwargs_model.get('source_redshift_list', None)
    
    @source_redshift_list.setter
    def source_redshift_list(self, value):
        self.kwargs_model['source_redshift_list'] = value

    @property
    def lens_model(self):
        return LensModel(self.lens_model_list, use_jax=self.use_jax)
    
    @property
    def lens_light_model(self):
        return LightModel(self.lens_light_model_list)
    
    @property
    def source_light_model(self):
        return LightModel(self.source_light_model_list)

    # @property
    # def lens_model_macromodel(self):
    #     if self.lens_model_macromodel is None:
    #         raise ValueError("This value is only populated when a substructure realization is added.")
    #     return LensModel(self.lens_model_list_macromodel, use_jax=self.use_jax)

    # @property
    # def lens_model_list_macromodel(self):
    #     if self.lens_model_list_macromodel is None:
    #         raise ValueError("This value is only populated when a substructure realization is added.")
    #     return self.lens_model_list_macromodel

    # @property
    # def kwargs_lens_macromodel(self):
    #     if self.kwargs_lens_macromodel is None:
    #         raise ValueError("This value is only populated when a substructure realization is added.")
    #     return self.kwargs_lens_macromodel

    # @property
    # def lens_redshift_list_macromodel(self):
    #     if self.lens_redshift_list_macromodel is None:
    #         raise ValueError("This value is only populated when a substructure realization is added.")
    #     return self.lens_redshift_list_macromodel
    
    def __str__(self):
        return f"StrongLens(name={self.name}, coords={self.coords}, z_lens={getattr(self, 'z_lens', None)}, z_source={getattr(self, 'z_source', None)})"
