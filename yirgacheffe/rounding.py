from __future__ import annotations

import math
import sys

from .window import PixelScale

def almost_equal(aval: float, bval: float) -> bool:
    """Safe floating point equality check."""
    return abs(aval - bval) < sys.float_info.epsilon

# As per https://xkcd.com/2170/, we need to stop caring about floating point
# accuracy at some point as it becomes problematic.
# The value here is 1 meter, given that geo data that we've been working with
# is accurate to 100 meter, but if you need to worry about the biodiversity
# of a virus in a petri dish this assumption may not work for you.
MINIMAL_DISTANCE_OF_INTEREST = 1.0
DISTANCE_PER_DEGREE_AT_EQUATOR = 40075017 / 360
MINIMAL_DEGREE_OF_INTEREST = MINIMAL_DISTANCE_OF_INTEREST / DISTANCE_PER_DEGREE_AT_EQUATOR

def round_up_pixels(value: float, pixelscale: float) -> int:
    """In general we round up pixels, as we don't want to lose range data,
    but floating point math means we will get errors where the value of pixel
    scale value rounds us to a tiny faction of a pixel up, and so math.ceil
    would round us up for microns worth of distance. """
    floored = math.floor(value)
    diff = value - floored
    degrees_diff = diff * pixelscale
    if degrees_diff < MINIMAL_DEGREE_OF_INTEREST:
        return floored
    else:
        return math.ceil(value)

def round_down_pixels(value: float, pixelscale: float) -> int:
    ceiled = math.ceil(value)
    diff = ceiled - value
    degrees_diff = diff * pixelscale
    if degrees_diff < MINIMAL_DEGREE_OF_INTEREST:
        return ceiled
    else:
        return math.floor(value)

def are_pixel_scales_equal_enough(pixel_scales: list[PixelScale | None]) -> bool:
    # some layers (e.g., constant layers) have no scale, and always work, so filter
    # them out first
    cleaned_pixel_scales: list[PixelScale] = [x for x in pixel_scales if x is not None]

    try:
        first = cleaned_pixel_scales[0]
    except IndexError:
        # empty list is close enough to itself
        return True

    for other in cleaned_pixel_scales[1:]:
        if (abs(first.xstep - other.xstep) > MINIMAL_DEGREE_OF_INTEREST) or \
                (abs(first.ystep - other.ystep) > MINIMAL_DEGREE_OF_INTEREST):
            return False
    return True
