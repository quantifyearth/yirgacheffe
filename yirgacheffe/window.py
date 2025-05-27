import math
import sys
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional

PixelScale = namedtuple('PixelScale', ['xstep', 'ystep'])

@dataclass
class Area:
    left: float
    top: float
    right: float
    bottom: float

    def __hash__(self):
        return (self.left, self.top, self.right, self.bottom).__hash__()

    def __eq__(self, other) -> bool:
        return math.isclose(self.left, other.left, abs_tol=1e-09) and \
            math.isclose(self.right, other.right, abs_tol=1e-09) and \
            math.isclose(self.top, other.top, abs_tol=1e-09) and \
            math.isclose(self.bottom, other.bottom, abs_tol=1e-09)

    def grow(self, offset: float):
        return Area(
            left=self.left - offset,
            top=self.top + offset,
            right=self.right + offset,
            bottom=self.bottom - offset
        )

    def overlaps(self, other) -> bool:
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

    def grow(self, pixels: int):
        return Window(
            xoff=self.xoff - pixels,
            yoff=self.xoff - pixels,
            xsize=self.xsize + (2 * pixels),
            ysize=self.ysize + (2 * pixels),
        )

    @staticmethod
    def find_intersection(windows: List) -> "Window":
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
    def find_intersection_no_throw(windows: List) -> Optional["Window"]:
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
