import sys

from dataclasses import dataclass
from typing import List

@dataclass
class Area:
    left: float
    top: float
    right: float
    bottom: float

    def grow(self, offset: float):
        return Area(
            left=self.left - offset,
            top=self.top + offset,
            right=self.right + offset,
            bottom=self.bottom - offset
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
            if window.xoff > left:
                left = window.xoff
            if window.yoff > top:
                top = window.yoff
            w_right = window.xoff + window.xsize
            if  w_right < right:
                right = w_right
            w_bottom = window.yoff + window.ysize
            if w_bottom < bottom:
                bottom = w_bottom
        if (left >= right) or (top >= bottom):
            raise ValueError('No intersection possible')
        return Window(
            left,
            top,
            right - left,
            bottom - top
        )
