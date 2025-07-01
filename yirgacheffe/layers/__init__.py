from osgeo import ogr

from ..window import PixelScale
from .base import YirgacheffeLayer
from .rasters import RasterLayer, InvalidRasterBand
from .rescaled import RescaledRasterLayer
from .vectors import RasteredVectorLayer, VectorLayer
from .area import UniformAreaLayer
from .constant import ConstantLayer
from .group import GroupLayer, TiledGroupLayer
try:
    from .h3layer import H3CellLayer
except ModuleNotFoundError:
    pass

class Layer(RasterLayer):
    """A place holder for now, at some point I want to replace Layer with RasterLayer."""


class VectorRangeLayer(RasteredVectorLayer):
    """Deprecated older name for VectorLayer"""

    def __init__(self, range_vectors: str, where_filter: str, scale: PixelScale, projection: str):
        vectors = ogr.Open(range_vectors)
        if vectors is None:
            raise FileNotFoundError(range_vectors)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)
        super().__init__(layer, scale, projection)


class DynamicVectorRangeLayer(VectorLayer):
    """Deprecated older name DynamicVectorLayer"""

    def __init__(self, range_vectors: str, where_filter: str, scale: PixelScale, projection: str):
        vectors = ogr.Open(range_vectors)
        if vectors is None:
            raise FileNotFoundError(range_vectors)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)
        super().__init__(layer, scale, projection)
