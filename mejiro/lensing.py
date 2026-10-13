"""Explicit forward-model backend selection for strong-lens simulations."""

from functools import lru_cache
from types import SimpleNamespace


def validate_backend(name):
    if not isinstance(name, str) or name not in {'lenstronomy', 'chalcedon'}:
        raise ValueError("backend must be 'lenstronomy' or 'chalcedon'")
    return name


@lru_cache(maxsize=2)
def get_backend(name):
    validate_backend(name)
    if name == 'chalcedon':
        from chalcedon import LensModel, LightModel, LensCosmo, LensEquationSolver
        from chalcedon.imaging import Coordinates, PixelGrid, PSF, ImageModel, PointSource, data_util, util
    else:
        from lenstronomy.Cosmo.lens_cosmo import LensCosmo
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        from lenstronomy.Data.coord_transforms import Coordinates
        from lenstronomy.Data.pixel_grid import PixelGrid
        from lenstronomy.Data.psf import PSF
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.PointSource.point_source import PointSource
        from lenstronomy.Util import data_util, util
    return SimpleNamespace(LensModel=LensModel, LightModel=LightModel, LensCosmo=LensCosmo,
                           LensEquationSolver=LensEquationSolver, Coordinates=Coordinates,
                           PixelGrid=PixelGrid, PSF=PSF, ImageModel=ImageModel,
                           PointSource=PointSource, data_util=data_util, util=util)