from typing import Any, Union

from ..operators import DataType
from ..window import Area, PixelScale, Window
from .base import YirgacheffeLayer
from .._backends import backend
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
        self.value = float(value)

    @property
    def datatype(self) -> DataType:
        return DataType.Float64

    def check_pixel_scale(self, _scale: PixelScale) -> bool:
        return True

    def set_window_for_intersection(self, _intersection: Area) -> None:
        pass

    def read_array(self, _x: int, _y: int, width: int, height: int) -> Any:
        return backend.full((height, width), self.value)

    def read_array_with_window(self, _x: int, _y: int, width: int, height: int, _window: Window) -> Any:
        return backend.full((height, width), self.value)

    def read_array_for_area(self, _target_area: Area, x: int, y: int, width: int, height: int) -> Any:
        return self.read_array(x, y, width, height)
