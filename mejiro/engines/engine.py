from abc import ABC, abstractmethod


class Engine(ABC):
    """Abstract base class for all simulation engines.

    Subclasses must implement `defaults` and `validate_engine_params` for
    each instrument they support.
    """

    def __init__(
            self
    ):
        pass

    @abstractmethod
    def defaults(self, instrument_name):
        """Return default engine parameters for the given instrument.

        Parameters
        ----------
        instrument_name : str
            Name of the instrument for which to retrieve defaults.

        Returns
        -------
        dict
            Mapping of parameter names to their default values.

        Raises
        ------
        NotImplementedError
            If the instrument is not supported by this engine.
        """
        pass

    @abstractmethod
    def validate_engine_params(self, instrument_name, engine_params):
        """Validate engine parameters for the given instrument.

        Parameters
        ----------
        instrument_name : str
            Name of the instrument whose parameters are being validated.
        engine_params : dict
            Mapping of parameter names to values to validate.

        Raises
        ------
        NotImplementedError
            If the instrument is not supported by this engine.
        ValueError
            If any parameter value is invalid for the given instrument.
        """
        pass

    @staticmethod
    def instrument_not_supported(instrument_name):
        """Raise an error if the instrument is not supported.

        Parameters
        ----------
        instrument_name : str
            Name of the unsupported instrument.

        Raises
        ------
        NotImplementedError
            Always raised to indicate the instrument is not supported.
        """
        raise NotImplementedError(f"{instrument_name} is not supported by this engine.")
