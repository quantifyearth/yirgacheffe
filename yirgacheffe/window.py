from collections import namedtuple
from dataclasses import dataclass
from typing import List

Area = namedtuple('Area', ['left', 'top', 'right', 'bottom'])

@dataclass
class Window:
    xoff: int
    yoff: int
    xsize: int
    ysize: int

    @property
    def as_array_args(self):
        return (self.xoff, self.yoff, self.xsize, self.ysize)

    def __lt__(self, other) -> bool:
        return (self.xsize < other.xsize) and \
            (self.ysize < other.ysize) and \
            (self.xoff >= other.xoff) and \
            (self.yoff >= other.yoff) and \
            ((self.xoff + self.xsize) < (other.xoff + other.xsize)) and \
            ((self.yoff + self.ysize) < (other.yoff + other.ysize))

    def __gt__(self, other) -> bool:
        return (self.xsize > other.xsize) and \
            (self.ysize > other.ysize) and \
            (self.xoff <= other.xoff) and \
            (self.yoff <= other.yoff) and \
            ((self.xoff + self.xsize) > (other.xoff + other.xsize)) and \
            ((self.yoff + self.ysize) > (other.yoff + other.ysize))

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

    @staticmethod
    def find_intersection(windows: List) -> "Window":
        if not windows:
            raise ValueError("Expected list of windows")
        areas = [Area(x.xoff, x.yoff, x.xoff + x.xsize, x.yoff + x.ysize) for x in windows]
        intersection = Area(
            left=max(x.left for x in areas),
            top=max(x.top for x in areas),
            right=min(x.right for x in areas),
            bottom=min(x.bottom for x in areas)
        )
        if (intersection.left >= intersection.right) or (intersection.top >= intersection.bottom):
            raise ValueError('No intersection possible')
        return Window(
            intersection.left,
            intersection.top,
            intersection.right - intersection.left,
            intersection.bottom - intersection.top
        )
