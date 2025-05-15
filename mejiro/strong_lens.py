from abc import ABC, abstractmethod
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel


class StrongLens(ABC):

    def __init__(
            self,
            name,
            coords,
            kwargs_model,
            kwargs_params,
            physical_params={}
    ):
        self.name = name
        self.coords = coords
        self.kwargs_model = kwargs_model
        self.kwargs_params = kwargs_params
        self.physical_params = physical_params

        # set cosmology
        if 'cosmo' in kwargs_model:
            self.cosmo = kwargs_model['cosmo']
        else:
            raise ValueError("Set astropy.cosmology instance in kwargs_model['cosmo']")

        # fields to initialize: these can be computed on demand
        self.lens_cosmo = None
        self.realization = None

    def add_realization(self, realization):
        """
        Add a pyHalo dark matter subhalo realization to the mass model of the system. See the [pyHalo documentation](https://github.com/dangilman/pyHalo) for details.

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
        return LensModel(self.lens_model_list)
    
    @property
    def lens_light_model(self):
        return LightModel(self.lens_light_model_list)
    
    @property
    def source_light_model(self):
        return LightModel(self.source_light_model_list)
