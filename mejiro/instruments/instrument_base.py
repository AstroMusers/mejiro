from abc import ABC, abstractmethod


class InstrumentBase(ABC):

    def __init__(
            self,
            name,
            bands,
            engines
    ):
        self.name = name
        self.bands = bands
        self.engines = engines

    @abstractmethod
    def get_pixel_scale(self, band):
        pass

    @abstractmethod
    def validate_instrument_params(params):
        pass

    @abstractmethod
    def default_params():
        pass
