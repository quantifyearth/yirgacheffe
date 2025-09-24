from __future__ import annotations
from math import floor, ceil
from pathlib import Path
from typing import Any

from skimage import transform

from ..window import MapProjection, PixelScale, Window
from .rasters import RasterLayer, YirgacheffeLayer
from .._backends import backend
from .._backends.enumeration import dtype as DataType


class RescaledRasterLayer(YirgacheffeLayer):
    """RescaledRaster dynamically rescales a raster, so to you to work with multiple layers at
    different scales without having to store unnecessary data. """

    @classmethod
    def layer_from_file(
        cls,
        filename: Path | str,
        pixel_scale: PixelScale,
        band: int = 1,
        nearest_neighbour: bool = True,
    ) -> RescaledRasterLayer:
        src = RasterLayer.layer_from_file(filename, band=band)
        source_projection = src.map_projection
        if source_projection is None:
            raise ValueError("Source raster must have projection and scale")
        target_projection = MapProjection(source_projection.name, pixel_scale.xstep, pixel_scale.ystep)
        return RescaledRasterLayer(src, target_projection, nearest_neighbour, src.name)

    def __init__(
        self,
        src: RasterLayer,
        target_projection: MapProjection,
        nearest_neighbour: bool = True,
        name: str | None = None,
    ):
        super().__init__(
            src.area,
            target_projection,
            name=name
        )

        self._src = src
        self._nearest_neighbour = nearest_neighbour

        src_projection = src.map_projection
        assert src_projection # from raster we should always have one

        self._x_scale = src_projection.xstep / target_projection.xstep
        self._y_scale = src_projection.ystep / target_projection.ystep

    def close(self):
        self._src.close()

    def _park(self):
        self._src._park()

    def _unpark(self):
        self._src._unpark()

    @property
    def datatype(self) -> DataType:
        return self._src.datatype

    def _read_array_with_window(
        self,
        xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:

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
