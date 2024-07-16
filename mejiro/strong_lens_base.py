from abc import ABC, abstractmethod


class StrongLensBase(ABC):

    def __init__(
            self,
            kwargs_model,
            kwargs_params,
            lens_mags,
            source_mags
            ):
        self.kwargs_model = kwargs_model
        self.kwargs_params = kwargs_params
        self.lens_mags = lens_mags
        self.source_mags = source_mags

        # TODO parse kwargs_model and kwargs_params

        # TODO set kwargs_lens, kwargs_lens_light, kwargs_source, kwargs_ps


    @abstractmethod
    def get_lens_mag(self, band):
        # TODO docstring
        pass

    @abstractmethod
    def get_source_mag(self, band):
        # TODO docstring
        pass

    @abstractmethod
    def get_einstein_radius(self):
        # TODO docstring
        # TODO theta_E may not be a key, so if it doesn't exist, compute it
        return self.kwargs_lens[0]['theta_E']
    