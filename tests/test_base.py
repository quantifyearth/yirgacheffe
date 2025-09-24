import math

import pytest

from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import YirgacheffeLayer
from yirgacheffe.window import Area, MapProjection

def test_pixel_to_latlng_unsupported_projection() -> None:
    layer = YirgacheffeLayer(
        Area(-10, 10, 10, -10),
        MapProjection("OTHER PROJECTION", 0.02, -0.02),
    )
    with pytest.raises(NotImplementedError):
        _ = layer.latlng_for_pixel(10, 10)

def test_pixel_from_latlng_unsupported_projection() -> None:
    layer = YirgacheffeLayer(
        Area(-10, 10, 10, -10),
        MapProjection("OTHER PROJECTION", 0.02, -0.02),
    )
    with pytest.raises(NotImplementedError):
        _ = layer.pixel_for_latlng(10.0, 10.0)

@pytest.mark.parametrize(
    "area,pixel,expected",
    [
        (
            Area(-10, 10, 10, -10),
            (0, 0),
            (10.0, -10.0)
        ),
        (
            Area(-10, 10, 10, -10),
            (1, 1),
            (9.8, -9.8)
        ),
        (
            Area(-10, 10, 10, -10),
            (101, 101),
            (-10.2, 10.2)
        ),
        (
            Area(-10, 10, 10, -10),
            (-1, -1),
            (10.2, -10.2)
        ),
        (
            Area(10, 10, 20, -10),
            (1, 1),
            (9.8, 10.2)
        ),
        (
            Area(-10, -10, 10, -20),
            (1, 1),
            (-10.2, -9.8)
        ),
    ]
)
def test_latlng_for_pixel(area: Area, pixel: tuple[int, int], expected: tuple[float, float]) -> None:
    layer = YirgacheffeLayer(
        area,
        MapProjection(WGS_84_PROJECTION, 0.2, -0.2),
    )
    result = layer.latlng_for_pixel(*pixel)
    assert math.isclose(result[0], expected[0])
    assert math.isclose(result[1], expected[1])

@pytest.mark.parametrize(
    "area,coord,expected",
    [
        (
            Area(-10, 10, 10, -10),
            (10.0, -10.0),
            (0, 0)
        ),
        (
            Area(-10, 10, 10, -10),
            (9.8, -9.8),
            (1, 1)
        ),
        (
            Area(-10, 10, 10, -10),
            (0.0, 0.0),
            (50, 50)
        ),
    ]
)
def test_pixel_for_latlng(area: Area, coord: tuple[float, float], expected: tuple[int, int]) -> None:
    layer = YirgacheffeLayer(
        area,
        MapProjection(WGS_84_PROJECTION, 0.2, -0.2),
    )
    result = layer.pixel_for_latlng(*coord)
    assert result == expected


@pytest.mark.parametrize(
    "area,window,pixel,expected",
    [
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (0, 0),
            (5.0, -5.0)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (1, 1),
            (4.8, -4.8)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (101, 101),
            (-15.2, 15.2)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (-1, -1),
            (5.2, -5.2)
        ),
        (
            Area(10, 10, 20, -10),
            Area(15, 5, 20, -5),
            (1, 1),
            (4.8, 15.2)
        ),
        (
            Area(-10, -10, 10, -20),
            Area(-5, -15, 5, -20),
            (1, 1),
            (-15.2, -4.8)
        ),
    ]
)
def test_latlng_for_pixel_with_intersection(
    area: Area,
    window: Area,
    pixel: tuple[int, int],
    expected: tuple[float, float]
) -> None:
    layer = YirgacheffeLayer(
        area,
        MapProjection(WGS_84_PROJECTION, 0.2, -0.2),
    )
    layer.set_window_for_intersection(window)
    result = layer.latlng_for_pixel(*pixel)
    assert math.isclose(result[0], expected[0])
    assert math.isclose(result[1], expected[1])

@pytest.mark.parametrize(
    "area,window,coord,expected",
    [
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (5.0, -5.0),
            (0, 0)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (4.8, -4.8),
            (1, 1)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            (0.0, 0.0),
            (25, 25)
        ),
    ]
)
def test_pixel_for_latlng_with_intersection(
    area: Area,
    window: Area,
    coord: tuple[float, float],
    expected: tuple[int, int]
) -> None:
    layer = YirgacheffeLayer(
        area,
        MapProjection(WGS_84_PROJECTION, 0.2, -0.2),
    )
    layer.set_window_for_intersection(window)
    result = layer.pixel_for_latlng(*coord)
    assert result == expected
