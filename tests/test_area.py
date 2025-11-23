import math

import pytest

from yirgacheffe import Area, MapProjection

@pytest.mark.parametrize(
    "lhs,rhs,is_equal,overlaps",
    [
        # Obvious equality
        (Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0), True,  True),
        (Area(-9.0, 9.0, 9.0, -9.0),     Area(-10.0, 10.0, 10.0, -10.0), False, True), # subset
        (Area(-9.0, 9.0, -1.0, 1.0),     Area(1.0, -1.0, 9.0, -9.0),     False, False),
        (Area(-10.0, 10.0, 1.0, -10.0),  Area(-1.0, 10.0, 10.0, -10.0),  False, True),
    ]
)
def test_area_operators(lhs: Area, rhs: Area, is_equal: bool, overlaps: bool) -> None:
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)
    assert (lhs.overlaps(rhs)) == overlaps
    assert (rhs.overlaps(lhs)) == overlaps
    assert not lhs.is_world
    assert not rhs.is_world

@pytest.mark.parametrize(
        "lhs,rhs,expected",
        [
            # Obvious equality
            (Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0)),
            (Area(-9.0, 9.0, 9.0, -9.0),     Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0)), # subset
            (Area(-9.0, 9.0, -1.0, 1.0),     Area(1.0, -1.0, 9.0, -9.0),     Area(-9.0, 9.0, 9.0, -9.0)),
            (Area(-10.0, 10.0, 10.0, -10.0), Area.world(),                   Area.world()),
            (Area.world(),                   Area(-10.0, 10.0, 10.0, -10.0), Area.world()),
        ]
    )
def test_area_union(lhs: Area, rhs: Area, expected: Area) -> None:
    union = lhs | rhs
    assert union == expected

@pytest.mark.parametrize(
        "lhs,rhs,expected",
        [
            # Obvious equality
            (Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0)),
            (Area(-9.0, 9.0, 9.0, -9.0),     Area(-10.0, 10.0, 10.0, -10.0), Area(-9.0, 9.0, 9.0, -9.0)), # subset
            (Area(-9.0, 9.0, -1.0, 1.0),     Area(1.0, -1.0, 9.0, -9.0),     None),
            (Area(-10.0, 10.0, 10.0, -10.0), Area.world(),                   Area(-10.0, 10.0, 10.0, -10.0)),
            (Area.world(),                   Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0)),
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
            Area(-10.1, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (-0.1, 0.0),
        ),
        (
            Area(10.1, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.1, 0.0),
        ),
        (
            Area(-9.9, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.1, 0.0),
        ),
        (
            Area(9.9, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (-0.1, 0.0),
        ),
        # Y shifted from perfect
        (
            Area(10.0, -10.1, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, -0.1),
        ),
        (
            Area(10.0, 10.1, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, 0.1),
        ),
        (
            Area(10.0, -9.9, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
            (0.0, 0.1),
        ),
        (
            Area(10.0, 9.9, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
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
