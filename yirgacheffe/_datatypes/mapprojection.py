from __future__ import annotations

import pyproj

from .pixelscale import PixelScale, are_pixel_scales_equal_enough

class MapProjection:
    """Records the map projection and the size of the pixels in a layer.

    This superceeeds the old PixelScale class, which will be removed in version 2.0.

    Args:
        name: The map projection used in WKT format, or as "epsg:xxxx" or "esri:xxxx".
        xstep: The number of units horizontal distance a step of one pixel makes in the map projection.
        ystep: The number of units vertical distance a step of one pixel makes in the map projection.

    Attributes:
        name: The map projection used in WKT format.
        xstep: The number of units horizontal distance a step of one pixel makes in the map projection.
        ystep: The number of units vertical distance a step of one pixel makes in the map projection.

    Examples:
        Create a map projection using an EPSG code:

        >>> proj_wgs84 = MapProjection("epsg:4326", 0.001, -0.001)

        Create a projection using an ESRI code:

        >>> proj_esri = MapProjection("esri:54030", 1000, -1000)
    """

    def __init__(self, projection_string: str, xstep: float, ystep: float) -> None:
        try:
            self.crs = pyproj.CRS.from_string(projection_string)
        except pyproj.exceptions.CRSError as exc:
            raise ValueError(f"Invalid projection: {projection_string}") from exc
        self.xstep = xstep
        self.ystep = ystep

    def __hash__(self):
        return hash((self.name, self.xstep, self.ystep))

    def __eq__(self, other) -> bool:
        if other is None:
            return True
        return (self.crs == other.crs) and \
            are_pixel_scales_equal_enough([self.scale, other.scale])

    @property
    def name(self) -> str:
        return self.crs.to_wkt()

    @property
    def epsg(self) -> int | None:
        return self.crs.to_epsg()

    @property
    def scale(self) -> PixelScale:
        return PixelScale(self.xstep, self.ystep)
