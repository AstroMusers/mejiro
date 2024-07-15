from abc import ABC, abstractmethod


class InstrumentBase(ABC):

    def __init__(
            self,
            name
            ):
        self.name = name


    @abstractmethod
    def psf_fwhm(self, band):
        pass
    