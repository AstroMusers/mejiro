from abc import ABC, abstractmethod


class StrongLens(ABC):

    def __init__(
            self,
            name,
            coords,
            kwargs_model,
            kwargs_params
    ):
        self.name = name
        self.coords = coords
        self.kwargs_model = kwargs_model
        self.kwargs_params = kwargs_params
