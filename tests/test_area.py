import math

import pytest

from yirgacheffe import Area, MapProjection

@pytest.mark.parametrize(
    "lhs,rhs,is_equal,overlaps",
    [
        # No projections
        ( # Obvious equality
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            True,
            True,
        ),
        ( # subset
            Area(-9.0, 9.0, 9.0, -9.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            False,
            True,
        ),
        (
            Area(-9.0, 9.0, -1.0, 1.0),
            Area(1.0, -1.0, 9.0, -9.0),
            False,
            False
        ),
        (
            Area(-10.0, 10.0, 1.0, -10.0),
            Area(-1.0, 10.0, 10.0, -10.0),
            False,
            True
        ),
        # Same projection equal offset
        ( # Obvious equality
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            True,
            True,
        ),
        ( # subset
            Area(-9.0, 9.0, 9.0, -9.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            False,
            True,
        ),
        (
            Area(-9.0, 9.0, -1.0, 1.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(1.0, -1.0, 9.0, -9.0, MapProjection("epsg:4326", 1.0, -1.0)),
            False,
            False
        ),
        (
            Area(-10.0, 10.0, 1.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-1.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            False,
            True
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, None),
            False,
            True,
        ),
        ( # Equal if we allow for grid offset wobble on one side
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.1, 10.0,  9.9, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            True,
            True,
        ),
        ( # Equal if we allow for grid offset wobble on one side
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.1, 10.0,  -9.9, MapProjection("epsg:4326", 1.0, -1.0)),
            True,
            True,
        ),
        ( # Equal if we allow for grid offset wobble on both sides
            Area( -9.9, 10.0, 10.1, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.1, 10.0,  9.9, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            True,
            True,
        ),
        ( # Equal if we allow for grid offset wobble on both sides
            Area(-10.0,  9.9, 10.0, -10.1, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.1, 10.0,  -9.9, MapProjection("epsg:4326", 1.0, -1.0)),
            True,
            True,
        ),
        ( # Not if we allow for larger grid offset wobble on both sides
            Area(-10.0,  9.6, 10.0, -10.4, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.6, 10.0,  -9.4, MapProjection("epsg:4326", 1.0, -1.0)),
            False,
            True,
        ),
    ]
)
def test_area_operators(lhs: Area, rhs: Area, is_equal: bool, overlaps: bool) -> None:
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)
    assert (lhs.overlaps(rhs)) == overlaps
    assert (rhs.overlaps(lhs)) == overlaps
    assert not lhs.is_world
    assert not rhs.is_world

def test_area_operators_mixed_projections() -> None:
    lhs = Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0))
    rhs = Area(-100.0, 100.0, 100.0, -100.0, MapProjection("esri:54009", 100, -100))
    assert (lhs == rhs) == False
    assert (lhs != rhs) == True
    with pytest.raises(ValueError):
        _ = lhs.overlaps(rhs)
    with pytest.raises(ValueError):
        _ = lhs | rhs
    with pytest.raises(ValueError):
        _ = lhs & rhs

@pytest.mark.parametrize(
    "lhs,rhs,expected",
    [
        ( # Obvious equality
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0)
        ),
        ( # subset
            Area(-9.0, 9.0, 9.0, -9.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0)
        ),
        (
            Area(-9.0, 9.0, -1.0, 1.0),
            Area(1.0, -1.0, 9.0, -9.0),
            Area(-9.0, 9.0, 9.0, -9.0)
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area.world(),
            Area.world()
        ),
        (
            Area.world(),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area.world()
        ),
        ( # Obvious equality, mixed projection and non
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Obvious equality, same projection
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Expect the unprojected one to be rounded up
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.2, 10.2, 10.2, -10.2),
            Area(-11.0, 11.0, 11.0, -11.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Expect the unprojected one to be rounded up
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-9.8, 9.8, 9.8, -9.8),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Two different grid offsets on the same projection
            Area(-10.1, 10.0,  9.9, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area( -9.9, 10.0, 10.1, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
    ]
)
def test_area_union(lhs: Area, rhs: Area, expected: Area) -> None:
    union = lhs | rhs
    assert union == expected

@pytest.mark.parametrize(
    "lhs,rhs,expected",
    [
        # Obvious equality
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0),
        ),
        (
            Area(-9.0, 9.0, 9.0, -9.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-9.0, 9.0, 9.0, -9.0),
        ), # subset
        (
            Area(-9.0, 9.0, -1.0, 1.0),
            Area(1.0, -1.0, 9.0, -9.0),
            None,
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area.world(),
            Area(-10.0, 10.0, 10.0, -10.0),
        ),
        (
            Area.world(),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0)
        ),
        ( # Obvious equality
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Obvious equality
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.2, 10.2, 10.2, -10.2),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Expect the unprojected one to be rounded up
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-9.2, 9.2, 9.2, -9.2),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        ( # Two different grid offsets on the same projection
            Area(-10.1, 10.0,  9.9, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area( -9.9, 10.0, 10.1, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
    ]
)
def test_area_intersection(lhs: Area, rhs: Area, expected: Area | None) -> None:
    if expected is not None:
        intersection = lhs & rhs
        assert intersection == expected
    else:
        with pytest.raises(ValueError):
            _ = lhs & rhs

def test_global_area() -> None:
    area = Area.world()
    assert area.is_world

    other_area = Area(-10.0, 10.0, 10.0, -10.0)
    assert area.overlaps(other_area)
    assert other_area.overlaps(area)

def test_wrong_types_on_eq() -> None:
    assert Area(-10, 10, 10, -10) != 42

@pytest.mark.parametrize(
    "area,expected",
    [
        # No shift from "perfect"
        (
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, 0.0),
        ),
        # X shifted from perfect
        (
            Area(-10.1, 10.0, 9.9, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (-0.1, 0.0),
        ),
        (
            Area(10.1, 10.0, 11.1, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.1, 0.0),
        ),
        (
            Area(-9.9, 10.0, 10.1, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.1, 0.0),
        ),
        (
            Area(9.9, 10.0, 10.9, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (-0.1, 0.0),
        ),
        # Y shifted from perfect
        (
            Area(10.0, -10.1, 10.0, -11.1, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, -0.1),
        ),
        (
            Area(10.0, 10.1, 10.0, -9.9, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, 0.1),
        ),
        (
            Area(10.0, -9.9, 10.0, -10.9, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, 0.1),
        ),
        (
            Area(10.0, 9.9, 10.0, -10.1, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, -0.1),
        ),
        # No projection
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            None,
        ),
    ]
)
def test_grid_offset(area: Area, expected: tuple[float,float] | None) -> None:
    offset = area._grid_offset
    if expected is None:
        assert offset is None
    else:
        assert offset is not None
        assert math.isclose(offset[0], expected[0])
        assert math.isclose(offset[1], expected[1])

@pytest.mark.parametrize(
    "lhs,rhs,expected",
    [
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-5.0, 5.0, 5.0, -5.0, MapProjection("epsg:4326", 0.1, -0.1)),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 0.1, -0.1)),
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-5.01, 5.0, 4.99, -5.0, MapProjection("epsg:4326", 0.1, -0.1)),
            Area(-10.01, 10.0, 9.99, -10.0, MapProjection("epsg:4326", 0.1, -0.1)),
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-4.99, 5.0, 5.01, -5.0, MapProjection("epsg:4326", 0.1, -0.1)),
            Area(-10.01, 10.0, 9.99, -10.0, MapProjection("epsg:4326", 0.1, -0.1)),
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-5.0, 5.01, 5.0, -4.99, MapProjection("epsg:4326", 0.1, -0.1)),
            Area(-10.01, 10.0, 9.99, -10.0, MapProjection("epsg:4326", 0.1, -0.1)),
        ),
        (
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-5.0, 4.99, 5.0, -5.01, MapProjection("epsg:4326", 0.1, -0.1)),
            Area(-10.01, 10.0, 9.99, -10.0, MapProjection("epsg:4326", 0.1, -0.1)),
        ),
    ]
)
def test_project_like(lhs: Area, rhs: Area, expected: Area) -> None:
    assert lhs.project_like(rhs) == expected

@pytest.mark.parametrize("area_args", [
    (-5.01, 5.0, 5.0, -5.0, MapProjection("epsg:4326", 0.1, -0.1)),
    (-5.0, 5.01, 5.0, -5.0, MapProjection("epsg:4326", 0.1, -0.1)),
])
def test_invalid_projected_areas(area_args) -> None:
    with pytest.raises(ValueError):
        _ = Area(*area_args)
