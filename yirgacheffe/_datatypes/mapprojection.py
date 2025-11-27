from __future__ import annotations
import math

import pyproj
from pyproj import CRS, Transformer

from .pixelscale import PixelScale

# As per https://xkcd.com/2170/, we need to stop caring about floating point
# accuracy at some point as it becomes problematic.
# The value here is 1 meter, given that geo data that we've been working with
# is accurate to 100 meter, but if you need to worry about the biodiversity
# of a virus in a petri dish this assumption may not work for you.
#
# TODO: all these values are very much picked due to the global projects Yirgacheffe was
# originally developed as part of. Given we now see satellite image data down to tens of
# centimetres being procured, these values are questionable, and should probably be a
# fraction of the actual pixel scale.
MINIMAL_DISTANCE_OF_INTEREST_IN_M = 1.0
DISTANCE_PER_DEGREE_AT_EQUATOR = 40075017 / 360
MINIMAL_DEGREE_OF_INTEREST = MINIMAL_DISTANCE_OF_INTEREST_IN_M / DISTANCE_PER_DEGREE_AT_EQUATOR
MINIMAL_DISTANCE_CRS = CRS.from_string("epsg:4326")

class MapProjection:
    """Records the map projection and the size of the pixels in a layer.

    Note: It is very common to find that round errors creep into pixel scale values in
    GeoTIFFs from different sources, and so MapProjection tolerates small amounts of difference between
    pixel scales that are below a single metre in resolution.

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
            self.crs = CRS.from_string(projection_string)
        except pyproj.exceptions.CRSError as exc:
            raise ValueError(f"Invalid projection: {projection_string}") from exc
        self.xstep = xstep
        self.ystep = ystep

        transformer = Transformer.from_crs(MINIMAL_DISTANCE_CRS, self.crs, always_xy=True)
        self._min_x_step, self._min_y_step = transformer.transform(
            MINIMAL_DEGREE_OF_INTEREST,
            MINIMAL_DEGREE_OF_INTEREST,
        )

    def __repr__(self) -> str:
        return f"MapProjection({self.crs.to_string()!r}, {self.xstep}, {self.ystep})"

    def __hash__(self):
        return hash((self.name, self.xstep, self.ystep))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapProjection):
            return False
        if self.crs != other.crs:
            return False
        return (abs(self.xstep - other.xstep) < self._min_x_step) and \
            (abs(self.ystep - other.ystep) < self._min_y_step)

    @property
    def name(self) -> str:
        return self.crs.to_wkt()

    @property
    def epsg(self) -> int | None:
        return self.crs.to_epsg()

    @property
    def scale(self) -> PixelScale:
        return PixelScale(self.xstep, self.ystep)

    def round_up_pixels(self, x: float, y: float) -> tuple[int, int]:
        """In general we round up pixels, as we don't want to lose range data,
        but floating point math means we will get errors where the value of pixel
        scale value rounds us to a tiny faction of a pixel up, and so math.ceil
        would round us up for microns worth of distance. """
        floored_x = math.floor(x)
        floored_y = math.floor(y)
        diff_x = x - floored_x
        diff_y = y - floored_y
        spatial_diff_x = diff_x * abs(self.xstep)
        spatial_diff_y = diff_y * abs(self.ystep)

        final_x = floored_x if (spatial_diff_x < self._min_x_step) else math.ceil(x)
        final_y = floored_y if (spatial_diff_y < self._min_y_step) else math.ceil(y)

        return (final_x, final_y)

    def round_down_pixels(self, x: float, y: float) -> tuple[int, int]:
        ceiled_x = math.ceil(x)
        ceiled_y = math.ceil(y)
        diff_x = ceiled_x - x
        diff_y = ceiled_y - y
        spatial_diff_x = diff_x * abs(self.xstep)
        spatial_diff_y = diff_y * abs(self.ystep)

        final_x = ceiled_x if (spatial_diff_x < self._min_x_step) else math.floor(x)
        final_y = ceiled_y if (spatial_diff_y < self._min_y_step) else math.floor(y)

        return (final_x, final_y)
