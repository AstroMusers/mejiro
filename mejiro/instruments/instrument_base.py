from abc import ABC, abstractmethod


class InstrumentBase(ABC):

    def __init__(
            self,
            name
    ):
        self.name = name

    # TODO enforce bands an attribute

    # @abstractmethod
    # def get_pixel_scale(self, band):
    #     pass

    @abstractmethod
    def validate_instrument_config(self, config):
        pass
