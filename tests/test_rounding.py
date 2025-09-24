import sys

import pytest

from yirgacheffe.window import PixelScale
from yirgacheffe.rounding import almost_equal, are_pixel_scales_equal_enough, round_up_pixels,\
    round_down_pixels, MINIMAL_DEGREE_OF_INTEREST

@pytest.mark.parametrize("lval,rval,expected",
    [
        (1.0, 1.0, True),
        (1.0, 1.1, False),
        (sys.float_info.epsilon / 2, 0.0, True),
        (sys.float_info.epsilon * 2, 0.0, False),
    ]
)
def test_almost_equal(lval, rval, expected):
    assert almost_equal(lval, rval) == expected

# The pixel scale here comes from the jung dataset, which is 400752 pixles
# wide, or 100M per pixel at the equator roughly.
@pytest.mark.parametrize("pixels,scale,expected",
    [
        (8033.000000000001, 0.0008983152841195215, 8033), # actual seen value
        (8033.001, 0.0008983152841195215, 8033),          # obvious just below test case
        (8033.01, 0.0008983152841195215, 8034),           # obvious just above test case
        (8032.999999999999, 0.0008983152841195215, 8033), # actual seen value
    ]
)
def test_pixel_rounding_up(pixels: float, scale: float, expected: int) -> None:
    assert round_up_pixels(pixels, scale) == expected

@pytest.mark.parametrize("pixels,scale,expected",
    [
        (55.99999999999926, 0.0008983152841195215, 56), # actual seen value
        (55.998, 0.0008983152841195215, 56),            # obvious just below test case
        (55.98, 0.0008983152841195215, 55),             # obvious just above test case
        (55.000000000001, 0.0008983152841195215, 55),   # actual seen value
    ]
)
def test_pixel_rounding_down(pixels: float, scale: float, expected: int) -> None:
    assert round_down_pixels(pixels, scale) == expected

@pytest.mark.parametrize("pixel_scales,expected",
    [
        (
            [],
            True
        ),
        (
            [
                PixelScale(0.1, 0.1),
            ],
            True
        ),
        (
            [
                PixelScale(0.1, 0.1),
                None,
            ],
            True
        ),
        (
            [
                PixelScale(0.1, 0.1),
                PixelScale(0.1, 0.1),
            ],
            True
        ),
        (
            [
                PixelScale(0.1, 0.1),
                PixelScale(0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2), 0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2)),
            ],
            True
        ),
        (
            [
                PixelScale(0.1, 0.1),
                PixelScale(0.1 + (MINIMAL_DEGREE_OF_INTEREST * 2), 0.1 + (MINIMAL_DEGREE_OF_INTEREST * 2)),
            ],
            False
        ),
    ]
)
def test_pixel_scale_comparison(pixel_scales: list[PixelScale], expected: bool) -> None:
    assert are_pixel_scales_equal_enough(pixel_scales) == expected
