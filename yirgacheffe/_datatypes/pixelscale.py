from __future__ import annotations
import sys
from collections import namedtuple

PixelScale = namedtuple('PixelScale', ['xstep', 'ystep'])

def almost_equal(aval: float, bval: float) -> bool:
    """Safe floating point equality check."""
    return abs(aval - bval) < sys.float_info.epsilon
