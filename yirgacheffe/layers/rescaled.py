from math import floor, ceil
from typing import Any, Optional

from skimage import transform

from ..window import PixelScale
from .rasters import RasterLayer, YirgacheffeLayer


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
    ):
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

        self._x_scale = src._pixel_scale.xstep / pixel_scale.xstep
        self._y_scale = src._pixel_scale.ystep / pixel_scale.ystep

    def close(self):
        self._src.close()

    def _park(self):
        self._src._park()

    def _unpark(self):
        self._src._unpark()

    def read_array(self, xoffset, yoffset, xsize, ysize) -> Any:
        # to avoid aliasing issues, we try to scale to the nearest pixel
        # and recrop when scaling bigger

        xoffset = xoffset + self.window.xoff
        yoffset = yoffset + self.window.yoff

        src_x_offset = floor(xoffset / self._x_scale)
        src_y_offset = floor(yoffset / self._y_scale)

        diff_x = floor(((xoffset / self._x_scale) - src_x_offset) * self._x_scale)
        diff_y = floor(((yoffset / self._x_scale) - src_y_offset) * self._x_scale)

        src_x_width = ceil((xsize + diff_x) / self._x_scale)
        src_y_width = ceil((ysize + diff_y) / self._y_scale)

        # Get the matching src data
        src_data = self._src.read_array(
            src_x_offset,
            src_y_offset,
            src_x_width,
            src_y_width
        )

        scaled = transform.resize(
            src_data,
            (src_y_width * self._y_scale, src_x_width * self._x_scale),
            order=(0 if self._nearest_neighbour else 1),
            anti_aliasing=(not self._nearest_neighbour)
        )

        return scaled[diff_y:(diff_y + ysize),diff_x:(diff_x + xsize)]
