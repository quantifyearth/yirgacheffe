from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Area:
    """Class to hold a geospatial area of data in the given projection.

    You can use set operators | (union) and & (intersection) on Areas.

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
        if self.is_world and other.is_world:
            return True
        return math.isclose(self.left, other.left, abs_tol=1e-09) and \
            math.isclose(self.right, other.right, abs_tol=1e-09) and \
            math.isclose(self.top, other.top, abs_tol=1e-09) and \
            math.isclose(self.bottom, other.bottom, abs_tol=1e-09)

    def __and__(self, other: object) -> Area:
        # Set intersection
        if not isinstance(other, Area):
            raise  ValueError("Can only intersect two areas")
        if self.is_world:
            return other
        if other.is_world:
            return self
        all_areas = [self, other]
        intersection = Area(
            left=max(x.left for x in all_areas),
            top=min(x.top for x in all_areas),
            right=min(x.right for x in all_areas),
            bottom=max(x.bottom for x in all_areas)
        )
        if (intersection.left >= intersection.right) or (intersection.bottom >= intersection.top):
            raise ValueError('No intersection possible')
        return intersection

    def __or__(self, other: object) -> Area:
        # Set union
        if not isinstance(other, Area):
            raise  ValueError("Can only intersect two areas")
        if self.is_world:
            return self
        if other.is_world:
            return other
        all_areas = [self, other]
        union = Area(
            left=min(x.left for x in all_areas),
            top=max(x.top for x in all_areas),
            right=max(x.right for x in all_areas),
            bottom=min(x.bottom for x in all_areas)
        )
        return union

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
