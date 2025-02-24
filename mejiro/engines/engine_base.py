from abc import ABC, abstractmethod


class EngineBase(ABC):

    def __init__(
            self
    ):
        pass

    @abstractmethod
    def defaults(self, instrument_name):
        """
        Return default parameters for the engine.
        """
        pass

    @abstractmethod
    def validate_engine_params(self, instrument_name, engine_params):
        pass

    def instrument_not_supported(self, instrument_name):
        """
        Raise an error if the instrument is not supported.
        """
        raise NotImplementedError(f"{instrument_name} is not supported by this engine.")
