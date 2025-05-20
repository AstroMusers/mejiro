from abc import ABC, abstractmethod
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel


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

    def add_realization(self, realization, use_jax=True):
        """
        Add a pyHalo dark matter subhalo realization to the mass model of the system. See the `pyHalo documentation <https://github.com/dangilman/pyHalo>`__ for details.

        Parameters
        ----------
        realization : pyHalo realization object
            See the pyHalo documentation for details.
        """
        self.realization = realization

        # get lenstronomy lensing quantities
        halo_lens_model_list, halo_redshift_array, kwargs_halos, _ = realization.lensing_quantities(
            add_mass_sheet_correction=True)
        
        # halo_lens_model_list and kwargs_halos are lists, but halo_redshift_array is ndarray
        halo_redshift_list = list(halo_redshift_array)

        # add subhalos to lenstronomy objects that model the strong lens
        self.kwargs_lens += kwargs_halos
        self.lens_redshift_list += halo_redshift_list
        self.lens_model_list += halo_lens_model_list

        # set JAXtronomy flag
        if isinstance(self.use_jax, bool):
            self.use_jax = [self.use_jax]
        self.use_jax += [use_jax] * len(halo_lens_model_list) 
    
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
    
    def __str__(self):
        return f"StrongLens(name={self.name}, coords={self.coords}, z_lens={getattr(self, 'z_lens', None)}, z_source={getattr(self, 'z_source', None)})"
