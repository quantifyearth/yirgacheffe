from __future__ import annotations
import math
from dataclasses import dataclass

from .mapprojection import MapProjection

@dataclass(frozen=True)
class Area:
    """Class to hold a geospatial area. Can optionally have a projection associated.

    Ideally areas should always have a projection associated with them, however some data sources,
    notably polygon datasets like GeoJSON, do not store this, so we have to allow for projectionless
    areas.

    You can use set operators | (union) and & (intersection) on Areas.

    Args:
        left: Left most point in the projection space.
        top: Top most point in the projection space.
        right: Right most point in the projection space.
        bottom: Bottom most point in the projection space.
        projection: An optional map projection.

    Attributes:
        left: Left most point in the projection space.
        top: Top most point in the projection space.
        right: Right most point in the projection space.
        bottom: Bottom most point in the projection space.
        projection: An optional map projection.
    """
    left: float
    top: float
    right: float
    bottom: float
    projection: MapProjection | None = None

    def __post_init__(self) -> None:
        if self.projection is None:
            return

        # If we have a map projection then validate the dimensions are based on the pixel scale
        # Note we have to not use the obvious approach of using mod here, as floating point mod
        # is a bit iffy with values between 1 and 0
        # >>> 20 % 2
        # 0
        # >>> 20 % 1
        # 0
        # >>> 20 % 0.1
        # 0.0999999999999989
        # Note this is an FP issue not a Python issue AFAICT, I get the same behaviour in OCaml
        # for instance.
        width = self.right - self.left
        height = self.top - self.bottom
        x_pixels = abs(width / self.projection.xstep)
        y_pixels = abs(height / self.projection.ystep)

        if not math.isclose(x_pixels - round(x_pixels), 0.0, abs_tol=1e-09) or \
                not math.isclose(y_pixels - round(y_pixels), 0.0, abs_tol=1e-09):
            raise ValueError("Area expected to be an integer multiple of projection units")

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
        if self.is_world and other.is_world:
            return True
        if self.projection != other.projection:
            # raise ValueError("Cannot compare areas with different projections")
            return False

        self_offset = self._grid_offset
        other_offset = other._grid_offset

        if self_offset and other_offset:
            x_diff = self_offset[0] - other_offset[0]
            y_diff = self_offset[1] - other_offset[1]
        else:
            x_diff = 0.0
            y_diff = 0.0

        return \
            math.isclose(self.left, other.left + x_diff, abs_tol=1e-09) and \
            math.isclose(self.right, other.right + x_diff, abs_tol=1e-09) and \
            math.isclose(self.top, other.top + y_diff, abs_tol=1e-09) and \
            math.isclose(self.bottom, other.bottom + y_diff, abs_tol=1e-09)

    def __and__(self, other: object) -> Area:
        # Set intersection
        if not isinstance(other, Area):
            raise  ValueError("Can only intersect two areas")
        if self.is_world:
            return other
        if other.is_world:
            return self

        lhs = self
        rhs = other

        if lhs.projection is None and rhs.projection is not None:
            lhs = lhs.project_like(rhs)
        elif rhs.projection is None and lhs.projection is not None:
            rhs = rhs.project_like(lhs)

        if lhs.projection != rhs.projection:
            raise ValueError("Cannot intersect areas with different projections")

        # If we intersect two layers with different grid wobbles, then generate
        # a result that is aligned with the midpoint between them.
        lhs_offset = lhs._grid_offset
        rhs_offset = rhs._grid_offset
        if lhs_offset and rhs_offset:
            x_offset = (lhs_offset[0] - rhs_offset[0]) / 2
            y_offset = (lhs_offset[1] - rhs_offset[1]) / 2
        else:
            lhs_offset = (0.0, 0.0)
            rhs_offset = (0.0, 0.0)
            x_offset = 0.0
            y_offset = 0.0

        intersection = Area(
            left=max(lhs.left - lhs_offset[0], rhs.left - rhs_offset[0]) - x_offset,
            top=min(lhs.top - lhs_offset[1], rhs.top - rhs_offset[1]) - y_offset,
            right=min(lhs.right - lhs_offset[0], rhs.right - rhs_offset[0]) - x_offset,
            bottom=max(lhs.bottom - lhs_offset[1], rhs.bottom - rhs_offset[1]) - y_offset,
            projection=lhs.projection,
        )
        if (intersection.left >= intersection.right) or (intersection.bottom >= intersection.top):
            raise ValueError('No intersection possible')
        return intersection

    def __or__(self, other: object) -> Area:
        # Set union
        if not isinstance(other, Area):
            raise  ValueError("Can only union two areas")
        if self.is_world:
            return self
        if other.is_world:
            return other

        lhs = self
        rhs = other

        if lhs.projection is None and rhs.projection is not None:
            lhs = lhs.project_like(rhs)
        elif rhs.projection is None and lhs.projection is not None:
            rhs = rhs.project_like(lhs)

        if lhs.projection != rhs.projection:
            raise ValueError("Cannot union areas with different projections")

        # If we union two layers with different grid wobbles, then generate
        # a result that is aligned with the midpoint between them.
        lhs_offset = lhs._grid_offset
        rhs_offset = rhs._grid_offset
        if lhs_offset and rhs_offset:
            x_offset = (lhs_offset[0] - rhs_offset[0]) / 2
            y_offset = (lhs_offset[1] - rhs_offset[1]) / 2
        else:
            lhs_offset = (0.0, 0.0)
            rhs_offset = (0.0, 0.0)
            x_offset = 0.0
            y_offset = 0.0

        union = Area(
            left=min(lhs.left - lhs_offset[0], rhs.left - rhs_offset[0]) - x_offset,
            top=max(lhs.top - lhs_offset[1], rhs.top - rhs_offset[1]) - y_offset,
            right=max(lhs.right - lhs_offset[0], rhs.right - rhs_offset[0]) - x_offset,
            bottom=min(lhs.bottom - lhs_offset[1], rhs.bottom - rhs_offset[1]) - y_offset,
            projection=lhs.projection,
        )
        return union

    @property
    def _grid_offset(self) -> tuple[float,float] | None:
        if self.projection is None:
            return None

        abs_xstep = abs(self.projection.xstep)
        abs_ystep = abs(self.projection.ystep)

        xoff = self.left - ((self.left // abs_xstep) * abs_xstep)
        yoff = self.top - ((self.top // abs_ystep) * abs_ystep)

        if xoff > (abs_xstep / 2):
            xoff -= abs_xstep
        if yoff > (abs_ystep / 2):
            yoff -= abs_ystep

        return xoff, yoff

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

        lhs = self
        rhs = other

        if lhs.projection is None and rhs.projection is not None:
            lhs = lhs.project_like(rhs)
        elif rhs.projection is None and lhs.projection is not None:
            rhs = rhs.project_like(lhs)

        if lhs.projection != rhs.projection:
            raise ValueError("Cannot compare areas with different projections")

        return (
            (lhs.left <= rhs.left <= lhs.right) or
            (lhs.left <= rhs.right <= lhs.right) or
            (rhs.left <= lhs.left <= rhs.right) or
            (rhs.left <= lhs.right <= rhs.right)
        ) and (
            (lhs.top >= rhs.top >= lhs.bottom) or
            (lhs.top >= rhs.bottom >= lhs.bottom) or
            (rhs.top >= lhs.top >= rhs.bottom) or
            (rhs.top >= lhs.bottom >= rhs.bottom)
        )

    def project_like(self, other: Area) -> Area:
        """Takes a projectionless area and maps it onto a map projection based on an existing area.

        Because map projections have pixel scales associated with them, the area may be expanded
        to ensure that the original area is within the bounds when mapped to the pixel space of the other area.

        Will raise an exception if this area already has a map projection set, or if the other area does not.

        Args:
            other: The other area to take the map projection from.

        Returns:
            A new area with the projection map.
        """
        if self.projection is not None:
            raise ValueError("Changing projection is not supported currently")
        if other.projection is None:
            raise ValueError("Like area must be have map projection")

        offset = other._grid_offset
        assert offset # We know this should be true due to the above guard, but pylint does not.
        x_off, y_off = offset
        abs_xstep = abs(other.projection.xstep)
        abs_ystep = abs(other.projection.ystep)

        return Area(
            left=(math.floor(self.left / abs_xstep) * abs_xstep) + x_off,
            top=(math.ceil(self.top / abs_ystep) * abs_ystep) + y_off,
            right=(math.ceil(self.right / abs_xstep) * abs_xstep) + x_off,
            bottom=(math.floor(self.bottom / abs_ystep) * abs_ystep) + y_off,
            projection=other.projection
        )
