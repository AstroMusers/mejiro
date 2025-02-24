from abc import ABC, abstractmethod


class Engine(ABC):

    def __init__(
            self
    ):
        pass

    @abstractmethod
    def defaults(self, instrument_name):
        pass

    @abstractmethod
    def validate_engine_params(self, instrument_name, engine_params):
        pass

    def instrument_not_supported(instrument_name):
        """
        Raise an error if the instrument is not supported.
        """
        raise NotImplementedError(f"{instrument_name} is not supported by this engine.")
