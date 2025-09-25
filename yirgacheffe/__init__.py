from pathlib import Path

from osgeo import gdal
import tomli as tomllib

try:
    from importlib import metadata
    __version__: str = metadata.version(__name__)
except ModuleNotFoundError:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    __version__ = pyproject_data["project"]["version"]

from ._core import read_raster, read_rasters, read_shape, read_shape_like, constant, read_narrow_raster
from .constants import WGS_84_PROJECTION
from .window import Area, MapProjection, Window
from ._backends.enumeration import dtype as DataType

gdal.UseExceptions()
