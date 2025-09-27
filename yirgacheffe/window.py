from __future__ import annotations
import math
import sys
from collections import namedtuple
from dataclasses import dataclass

import pyproj

PixelScale = namedtuple('PixelScale', ['xstep', 'ystep'])

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
    """

    def __init__(self, projection_string: str, xstep: float, ystep: float) -> None:
        try:
            self.crs = pyproj.CRS.from_string(projection_string)
        except pyproj.exceptions.CRSError as exc:
            raise ValueError(f"Invalid projection: {projection_string}") from exc
        self.xstep = xstep
        self.ystep = ystep

    def __eq__(self, other) -> bool:
        if other is None:
            return True
        # to avoid circular dependancies
        from .rounding import are_pixel_scales_equal_enough  # pylint: disable=C0415
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

@dataclass
class Area:
    """Class to hold a geospatial area of data in the given projection.

    Args:
        left: Left most point in the projection space.
        top: Top most point in the projection space.
        right: Right most point in the projection space.
        bottom: Bottom most point in the projection space.

    Attributes:
        left: Left most point in the projection space.
        top: Top most point in the projection space.
        right: Right most point in the projection space.
        bottom: Bottom most point in the projection space.
    """
    left: float
    top: float
    right: float
    bottom: float

    @staticmethod
    def world() -> Area:
        """Creates an area that covers the entire planet.

        Returns:
            An area where the extents are nan, but is_world returns true.
        """
        return Area(float("nan"), float("nan"), float("nan"), float("nan"))

    def __hash__(self):
        return (self.left, self.top, self.right, self.bottom).__hash__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Area):
            return False
        return math.isclose(self.left, other.left, abs_tol=1e-09) and \
            math.isclose(self.right, other.right, abs_tol=1e-09) and \
            math.isclose(self.top, other.top, abs_tol=1e-09) and \
            math.isclose(self.bottom, other.bottom, abs_tol=1e-09)

    def grow(self, offset: float) -> Area:
        """Expand the area in all directions by the given amount.

        Generates a new area that is an expanded version of the current area.

        Args:
            offset: The amount by which to grow the area.

        Returns:
            A new area of the expanded size.
        """
        return Area(
            left=self.left - offset,
            top=self.top + offset,
            right=self.right + offset,
            bottom=self.bottom - offset
        )

    @property
    def is_world(self) -> bool:
        """Returns true if this is a global area, independent of projection.

        Returns:
            True if the Area was created with `world` otherwise False.
        """
        return math.isnan(self.left)

    def overlaps(self, other: Area) -> bool:
        """Check if this area overlaps with another area.

        Args:
            other: The other area to compare this area with.

        Returns:
            True if the two areas intersect, otherwise false.
        """

        if self.is_world or other.is_world:
            return True

        return (
            (self.left <= other.left <= self.right) or
            (self.left <= other.right <= self.right) or
            (other.left <= self.left <= other.right) or
            (other.left <= self.right <= other.right)
        ) and (
            (self.top >= other.top >= self.bottom) or
            (self.top >= other.bottom >= self.bottom) or
            (other.top >= self.top >= other.bottom) or
            (other.top >= self.bottom >= other.bottom)
        )

@dataclass
class Window:
    """Class to hold the pixel dimensions of data in the given projection.

    Args:
        xoff: X axis offset.
        yoff: Y axis offset.
        xsize: Width of data in pixels.
        ysize: Height of data in pixels.

    Attributes:
        xoff: X axis offset.
        yoff: Y axis offset.
        xsize: Width of data in pixels.
        ysize: Height of data in pixels.
    """
    xoff: int
    yoff: int
    xsize: int
    ysize: int

    @property
    def as_array_args(self) -> tuple[int, ...]:
        """A tuple containing xoff, yoff, xsize, and ysize."""
        return (self.xoff, self.yoff, self.xsize, self.ysize)

    def __lt__(self, other) -> bool:
        return (self.xsize < other.xsize) and \
            (self.ysize < other.ysize) and \
            (self.xoff >= other.xoff) and \
            (self.yoff >= other.yoff) and \
            ((self.xoff + self.xsize) <= (other.xoff + other.xsize)) and \
            ((self.yoff + self.ysize) <= (other.yoff + other.ysize))

    def __gt__(self, other) -> bool:
        return (self.xsize > other.xsize) and \
            (self.ysize > other.ysize) and \
            (self.xoff <= other.xoff) and \
            (self.yoff <= other.yoff) and \
            ((self.xoff + self.xsize) >= (other.xoff + other.xsize)) and \
            ((self.yoff + self.ysize) >= (other.yoff + other.ysize))

    def __le__(self, other) -> bool:
        return (self.xsize <= other.xsize) and \
            (self.ysize <= other.ysize) and \
            (self.xoff >= other.xoff) and \
            (self.yoff >= other.yoff) and \
            ((self.xoff + self.xsize) <= (other.xoff + other.xsize)) and \
            ((self.yoff + self.ysize) <= (other.yoff + other.ysize))

    def __ge__(self, other) -> bool:
        return (self.xsize >= other.xsize) and \
            (self.ysize >= other.ysize) and \
            (self.xoff <= other.xoff) and \
            (self.yoff <= other.yoff) and \
            ((self.xoff + self.xsize) >= (other.xoff + other.xsize)) and \
            ((self.yoff + self.ysize) >= (other.yoff + other.ysize))

    def grow(self, pixels: int) -> Window:
        """Expand the area in all directions by the given amount.

        Generates a new window that is an expanded version of the current window.

        Args:
            pixels: The amount by which to grow the window in pixels.

        Returns:
            A new window of the expanded size.
        """
        return Window(
            xoff=self.xoff - pixels,
            yoff=self.xoff - pixels,
            xsize=self.xsize + (2 * pixels),
            ysize=self.ysize + (2 * pixels),
        )

    @staticmethod
    def find_intersection(windows: list) -> Window:
        if not windows:
            raise ValueError("Expected list of windows")
        # This isn't very pythonic, as it was originally written, but
        # this method gets called a lot (by every Layer.read_array), so not looping
        # over the window list multiple times halves the performance cost (as
        # measured with cProfile).
        left, top, right, bottom = -sys.maxsize, -sys.maxsize, sys.maxsize, sys.maxsize
        for window in windows:
            left = max(left, window.xoff)
            top = max(top, window.yoff)
            right = min(right, window.xoff + window.xsize)
            bottom = min(bottom, window.yoff + window.ysize)
        if (left >= right) or (top >= bottom):
            raise ValueError('No intersection possible')
        return Window(
            left,
            top,
            right - left,
            bottom - top
        )

    @staticmethod
    def find_intersection_no_throw(windows: list) -> Window | None:
        if not windows:
            raise ValueError("Expected list of windows")
        # This isn't very pythonic, as it was originally written, but
        # this method gets called a lot (by every Layer.read_array), so not looping
        # over the window list multiple times halves the performance cost (as
        # measured with cProfile).
        left, top, right, bottom = -sys.maxsize, -sys.maxsize, sys.maxsize, sys.maxsize
        for window in windows:
            left = max(left, window.xoff)
            top = max(top, window.yoff)
            right = min(right, window.xoff + window.xsize)
            bottom = min(bottom, window.yoff + window.ysize)
        if (left >= right) or (top >= bottom):
            return None
        return Window(
            left,
            top,
            right - left,
            bottom - top
        )
