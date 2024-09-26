from abc import ABC, abstractmethod


class EngineBase(ABC):

    def __init__(
            self
    ):
        pass

    @abstractmethod
    def validate_engine_params(self, config):
        pass
