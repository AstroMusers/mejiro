from mejiro.strong_lens_base import StrongLensBase


class StrongLens(StrongLensBase):
    
        def __init__(
                self,
                name,
                lens_mag,
                source_mag
                ):
            super().__init__(name=name)
            self.lens_mag = lens_mag
            self.source_mag = source_mag
    
        def get_lens_mag(self, band):
            return self.lens_mag
    
        def get_source_mag(self, band):
            return self.source_mag
        