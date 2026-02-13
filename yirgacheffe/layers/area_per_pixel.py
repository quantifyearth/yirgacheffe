from __future__ import annotations
import math
from typing import Any

import numpy as np
from pyproj import Transformer

from .._datatypes import Area, MapProjection, Window
from .base import YirgacheffeLayer
from .._backends import backend
from .._backends.enumeration import dtype as DataType

# Based on https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
def area_of_pixel(a: float, b: float, pixel_scale: tuple[float,float], center_lat: float) -> float:
    x_scale, y_scale = pixel_scale
    e = math.sqrt(1 - ((b / a) ** 2))
    area_list = []
    for f in [center_lat + (y_scale / 2), center_lat - (y_scale / 2)]:
        sin_of_f = math.sin(math.radians(f))
        zm = 1 - (e * sin_of_f)
        zp = 1 + (e * sin_of_f)
        area_list.append(
            math.pi * (b ** 2) * (
                (math.log(zp / zm) / (2 * e)) +
                (sin_of_f / (zp * zm))
            )
        )
    return abs((x_scale / 360.) * (area_list[0] - area_list[1]))

class AreaPerPixelLayer(YirgacheffeLayer):
    """This layer will dynamically generate pixels where the value is the area in square meters of that pixel in
    the given map projection and pixel scale.
    """

    def __init__(self, projection: MapProjection):
        if projection is None:
            raise ValueError("projection should not be None")
        if not isinstance(projection, MapProjection):
            raise TypeError("projection should be of type MapProjection")
        if projection.crs.area_of_use is None:
            raise ValueError("CRS for this map projection has no bounds")

        west, south, east, north = projection.crs.area_of_use.bounds

        x_scale = abs(projection.xstep)
        y_scale = abs(projection.ystep)

        if projection.crs.is_geographic:
            area = Area(
                left=math.floor(west / x_scale) * x_scale,
                top=math.ceil(north / y_scale) * y_scale,
                right=math.ceil(east / x_scale) * x_scale,
                bottom=math.floor(south / y_scale) * y_scale,
                projection=projection,
            )
        else:
            # For projections like Mollweide, the corners aren't valid extremes, so we test
            # a mix of corners and mid-points and take the extremes on each axis
            horizontal = [west, ((west + east) / 2), east]
            vertical = [north, ((south + north) / 2), south]

            points = []
            for x in horizontal:
                points.extend([(x, i) for i in vertical])

            transformer = Transformer.from_crs("epsg:4326", projection.crs, always_xy=True)
            xs, ys = transformer.transform(*zip(*points)) # type: ignore

            left = math.floor(min(xs) / x_scale) * x_scale
            right = math.ceil(max(xs) / x_scale) * x_scale
            bottom = math.floor(min(ys) / y_scale) * y_scale
            top = math.floor(max(ys) / y_scale) * y_scale
            area = Area(left=left, top=top, right=right, bottom=bottom, projection=projection)

        super().__init__(area, projection.crs.area_of_use.name)

        self._hash = hash(projection._gdal_projection)

    @property
    def _cse_hash(self) -> int | None:
        return self._hash

    @property
    def datatype(self):
        return DataType.Float32

    def _read_array_with_window(
        self,
        _xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:
        projection = self.map_projection
        if projection is None:
            raise RuntimeError("Area Per Pixel layer should always be projected")

        if projection.crs.is_geographic:
            ellipsoid = projection.crs.ellipsoid
            if ellipsoid is None:
                raise RuntimeError("Exected geographic CRS to have ellipsoid details")
            a = ellipsoid.semi_major_metre
            b = ellipsoid.semi_minor_metre

            offset = window.yoff + yoffset
            x_scale = projection.xstep
            y_scale = projection.ystep

            area_values = np.array([
                area_of_pixel(a, b, (x_scale, y_scale), self.area.top + ((offset + i + 0.5) * y_scale))
                for i in range(ysize)
            ])
            area_array = np.broadcast_to(area_values[:, np.newaxis], (ysize, xsize))
            return backend.promote(area_array)

        else:
            pixel_area = abs(projection.xstep * projection.ystep)
            return backend.full((ysize, xsize), pixel_area)
