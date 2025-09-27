import math
import os
import tempfile

import pytest

import yirgacheffe as yg
from yirgacheffe.layers import YirgacheffeLayer
from yirgacheffe.window import Area, MapProjection

from tests.helpers import make_vectors_with_id

def test_pixel_to_latlng_no_projection() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)
        with yg.read_shape(path) as layer:
            with pytest.raises(ValueError):
                _ = layer.latlng_for_pixel(10, 10)

def test_latlng_to_pixel_no_projection() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)
        with yg.read_shape(path) as layer:
            with pytest.raises(ValueError):
                _ = layer.pixel_for_latlng(10.0, 10.0)

@pytest.mark.parametrize(
    "area,projection,pixel,expected",
    [
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (0, 0),
            (10.0, -10.0)
        ),
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (1, 1),
            (9.8, -9.8)
        ),
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (101, 101),
            (-10.2, 10.2)
        ),
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (-1, -1),
            (10.2, -10.2)
        ),
        (
            Area(10, 10, 20, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (1, 1),
            (9.8, 10.2)
        ),
        (
            Area(-10, -10, 10, -20),
            MapProjection("epsg:4326", 0.2, -0.2),
            (1, 1),
            (-10.2, -9.8)
        ),
    ]
)
def test_latlng_for_pixel(
    area: Area,
    projection: MapProjection,
    pixel: tuple[int, int],
    expected: tuple[float, float]
) -> None:
    layer = YirgacheffeLayer(
        area,
        projection,
    )
    result = layer.latlng_for_pixel(*pixel)
    assert math.isclose(result[0], expected[0])
    assert math.isclose(result[1], expected[1])

@pytest.mark.parametrize(
    "area,projection,coord,expected",
    [
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (10.0, -10.0),
            (0, 0)
        ),
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (9.8, -9.8),
            (1, 1)
        ),
        (
            Area(-10, 10, 10, -10),
            MapProjection("epsg:4326", 0.2, -0.2),
            (0.0, 0.0),
            (50, 50)
        ),
    ]
)
def test_pixel_for_latlng(
    area: Area,
    projection: MapProjection,
    coord: tuple[float, float],
    expected: tuple[int, int]
) -> None:
    layer = YirgacheffeLayer(
        area,
        projection,
    )
    result = layer.pixel_for_latlng(*coord)
    assert result == expected

@pytest.mark.parametrize(
    "area,window,projection,pixel,expected",
    [
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (0, 0),
            (5.0, -5.0)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (1, 1),
            (4.8, -4.8)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (101, 101),
            (-15.2, 15.2)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (-1, -1),
            (5.2, -5.2)
        ),
        (
            Area(10, 10, 20, -10),
            Area(15, 5, 20, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (1, 1),
            (4.8, 15.2)
        ),
        (
            Area(-10, -10, 10, -20),
            Area(-5, -15, 5, -20),
            MapProjection("epsg:4326", 0.2, -0.2),
            (1, 1),
            (-15.2, -4.8)
        ),
    ]
)
def test_latlng_for_pixel_with_intersection(
    area: Area,
    window: Area,
    projection: MapProjection,
    pixel: tuple[int, int],
    expected: tuple[float, float]
) -> None:
    layer = YirgacheffeLayer(
        area,
        projection,
    )
    layer.set_window_for_intersection(window)
    result = layer.latlng_for_pixel(*pixel)
    assert math.isclose(result[0], expected[0])
    assert math.isclose(result[1], expected[1])

@pytest.mark.parametrize(
    "area,window,projection,coord,expected",
    [
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (5.0, -5.0),
            (0, 0)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (4.8, -4.8),
            (1, 1)
        ),
        (
            Area(-10, 10, 10, -10),
            Area(-5, 5, 5, -5),
            MapProjection("epsg:4326", 0.2, -0.2),
            (0.0, 0.0),
            (25, 25)
        ),
    ]
)
def test_pixel_for_latlng_with_intersection(
    area: Area,
    window: Area,
    projection: MapProjection,
    coord: tuple[float, float],
    expected: tuple[int, int]
) -> None:
    layer = YirgacheffeLayer(
        area,
        projection,
    )
    layer.set_window_for_intersection(window)
    result = layer.pixel_for_latlng(*coord)
    assert result == expected
