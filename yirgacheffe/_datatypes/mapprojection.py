from __future__ import annotations
import math
from functools import lru_cache

import pyproj
from pyproj import CRS

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

# It turns out that calls to CRS.to_epsg are quite slow - not a problem as a one
# of call, but in our unit tests we make and destroy thousands of rasters, each of
# which has to do a to_epsg call to get the projection for saving. We saw one test
# that makes 1600 rasters take 72 seconds without this fix, and 2 seconds with it.
@lru_cache(maxsize=128)
def _get_projection_string(provided_name: str) -> str:
    crs = CRS.from_string(provided_name)
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.to_wkt()

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
        self._gdal_projection = _get_projection_string(projection_string)

    @property
    def _min_step(self) -> float:
        unit_name = self.crs.axis_info[0].unit_name
        if unit_name in ('metre', 'meter', 'm'):
            return MINIMAL_DISTANCE_OF_INTEREST_IN_M
        elif unit_name in ('degree', 'degrees'):
            return MINIMAL_DEGREE_OF_INTEREST
        else:
            raise NotImplementedError(f"Unsupported unit: {unit_name}")

    def __repr__(self) -> str:
        return f"MapProjection({self.crs.to_string()!r}, {self.xstep}, {self.ystep})"

    def __hash__(self):
        return hash((self.name, self.xstep, self.ystep))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MapProjection):
            return False
        if self.crs != other.crs:
            return False
        return (abs(self.xstep - other.xstep) < self._min_step) and \
            (abs(self.ystep - other.ystep) < self._min_step)

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

        final_x = floored_x if (spatial_diff_x < self._min_step) else math.ceil(x)
        final_y = floored_y if (spatial_diff_y < self._min_step) else math.ceil(y)

        return (final_x, final_y)

    def round_down_pixels(self, x: float, y: float) -> tuple[int, int]:
        ceiled_x = math.ceil(x)
        ceiled_y = math.ceil(y)
        diff_x = ceiled_x - x
        diff_y = ceiled_y - y
        spatial_diff_x = diff_x * abs(self.xstep)
        spatial_diff_y = diff_y * abs(self.ystep)

        final_x = ceiled_x if (spatial_diff_x < self._min_step) else math.floor(x)
        final_y = ceiled_y if (spatial_diff_y < self._min_step) else math.floor(y)

        return (final_x, final_y)
