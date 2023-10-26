from typing import Any, Union

import numpy as np
from osgeo import gdal

from ..window import Area, PixelScale
from .base import YirgacheffeLayer
from .. import WGS_84_PROJECTION


class ConstantLayer(YirgacheffeLayer):
    """This is a layer that will return the identity value - can be used when an input layer is
    missing (e.g., area) without having the calculation full of branches."""
    def __init__(self, value: Union[int,float]): # pylint: disable=W0231
        area = Area(
            left = -180.0,
            top = 90.0,
            right = 180.0,
            bottom = -90.0
        )
        super().__init__(area, None, WGS_84_PROJECTION)
        self.value = value

    @property
    def datatype(self) -> int:
        return gdal.GDT_CFloat64

    def check_pixel_scale(self, _scale: PixelScale) -> bool:
        return True

    def set_window_for_intersection(self, _intersection: Area) -> None:
        pass

    def read_array(self, _x: int, _y: int, xsize: int, ysize: int) -> Any:
        return np.full((ysize, xsize), self.value)
