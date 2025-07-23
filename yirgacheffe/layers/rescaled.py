from __future__ import annotations
from math import floor, ceil
from typing import Any, Optional

from skimage import transform
from yirgacheffe.operators import DataType

from ..window import PixelScale, Window
from .rasters import RasterLayer, YirgacheffeLayer
from .._backends import backend


class RescaledRasterLayer(YirgacheffeLayer):
    """RescaledRaster dynamically rescales a raster, so to you to work with multiple layers at
    different scales without having to store unnecessary data. """

    @classmethod
    def layer_from_file(
        cls,
        filename: str,
        pixel_scale: PixelScale,
        band: int = 1,
        nearest_neighbour: bool = True,
    ) -> RescaledRasterLayer:
        src = RasterLayer.layer_from_file(filename, band=band)
        return RescaledRasterLayer(src, pixel_scale, nearest_neighbour, src.name)

    def __init__(
        self,
        src: RasterLayer,
        pixel_scale: PixelScale,
        nearest_neighbour: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(
            src.area,
            pixel_scale=pixel_scale,
            projection=src.projection,
            name=name
        )

        self._src = src
        self._nearest_neighbour = nearest_neighbour

        src_pixel_scale = src.pixel_scale
        assert src_pixel_scale # from raster we should always have one

        self._x_scale = src_pixel_scale.xstep / pixel_scale.xstep
        self._y_scale = src_pixel_scale.ystep / pixel_scale.ystep

    def close(self):
        self._src.close()

    def _park(self):
        self._src._park()

    def _unpark(self):
        self._src._unpark()

    @property
    def datatype(self) -> DataType:
        return self._src.datatype

    def read_array_with_window(self, xoffset: int, yoffset: int, xsize: int, ysize: int, window: Window) -> Any:

        # to avoid aliasing issues, we try to scale to the nearest pixel
        # and recrop when scaling bigger

        xoffset = xoffset + window.xoff
        yoffset = yoffset + window.yoff

        src_x_offset = floor(xoffset / self._x_scale)
        src_y_offset = floor(yoffset / self._y_scale)

        diff_x = floor(((xoffset / self._x_scale) - src_x_offset) * self._x_scale)
        diff_y = floor(((yoffset / self._x_scale) - src_y_offset) * self._x_scale)

        src_x_width = ceil((xsize + diff_x) / self._x_scale)
        src_y_width = ceil((ysize + diff_y) / self._y_scale)

        # Get the matching src data
        src_data = backend.demote_array(self._src.read_array(
            src_x_offset,
            src_y_offset,
            src_x_width,
            src_y_width
        ))

        scaled = transform.resize(
            src_data,
            (src_y_width * self._y_scale, src_x_width * self._x_scale),
            order=(0 if self._nearest_neighbour else 1),
            anti_aliasing=(not self._nearest_neighbour)
        )

        return backend.promote(scaled[diff_y:(diff_y + ysize),diff_x:(diff_x + xsize)])
