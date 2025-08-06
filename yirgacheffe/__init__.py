from osgeo import gdal
try:
    from importlib import metadata
    __version__ = metadata.version(__name__)
except ModuleNotFoundError:
    __version__ = "unknown"

from ._core import read_raster, read_rasters, read_shape, read_shape_like
from .constants import WGS_84_PROJECTION

gdal.UseExceptions()
