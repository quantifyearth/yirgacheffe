import pytest

from yirgacheffe import MapProjection
from yirgacheffe._datatypes.mapprojection import MINIMAL_DEGREE_OF_INTEREST, MINIMAL_DISTANCE_OF_INTEREST_IN_M

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
    projection = MapProjection("epsg:4326", scale, -scale)
    assert projection.round_up_pixels(pixels, pixels) == (expected, expected)

@pytest.mark.parametrize("pixels,scale,expected",
    [
        (55.99999999999926, 0.0008983152841195215, 56), # actual seen value
        (55.998, 0.0008983152841195215, 56),            # obvious just below test case
        (55.98, 0.0008983152841195215, 55),             # obvious just above test case
        (55.000000000001, 0.0008983152841195215, 55),   # actual seen value
    ]
)
def test_pixel_rounding_down(pixels: float, scale: float, expected: int) -> None:
    projection = MapProjection("epsg:4326", scale, -scale)
    assert projection.round_down_pixels(pixels, pixels) == (expected, expected)

@pytest.mark.parametrize("lhs,rhs,expected", [
    (
        MapProjection("epsg:4326", 0.1, 0.1),
        None,
        True,
    ),
    (
        MapProjection("epsg:4326", 0.1, 0.1),
        MapProjection("epsg:4326", 0.1, 0.1),
        True,
    ),
    (
        MapProjection("epsg:4326", 0.1, 0.1),
        MapProjection("epsg:4326", 0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2), 0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2)),
        True,
    ),
    (
        MapProjection("epsg:4326", 0.1, 0.1),
        MapProjection("epsg:4326", 0.1 + (MINIMAL_DEGREE_OF_INTEREST * 2), 0.1 + (MINIMAL_DEGREE_OF_INTEREST * 2)),
        False,
    ),
    (
        MapProjection("epsg:4326", 0.1, 0.1),
        MapProjection("esri:54009", 0.1, 0.1),
        False,
    ),
    (
        MapProjection("esri:54009", 100.0, 100.0),
        MapProjection("esri:54009", 100.0, 100.0),
        True,
    ),
    (
        # this test attempts to abuse the fact that ESRI:54009 is in metres
        MapProjection("esri:54009", 100.0, 100.0),
        MapProjection(
            "esri:54009",
            100.0 + (MINIMAL_DISTANCE_OF_INTEREST_IN_M / 2),
            100.0 + (MINIMAL_DISTANCE_OF_INTEREST_IN_M / 2),
        ),
        True,
    ),
    (
        # this test attempts to abuse the fact that ESRI:54009 is in metres
        MapProjection("esri:54009", 100.0, 100.0),
        MapProjection(
            "esri:54009",
            100.0 + (MINIMAL_DISTANCE_OF_INTEREST_IN_M * 2),
            100.0 + (MINIMAL_DISTANCE_OF_INTEREST_IN_M * 2),
        ),
        False,
    ),
])
def test_pixel_scale_comparison(lhs: MapProjection, rhs: MapProjection, expected: bool) -> None:
    assert (lhs == rhs) == expected
