import pytest

from yirgacheffe.window import MapProjection
from yirgacheffe.rounding import MINIMAL_DEGREE_OF_INTEREST

def test_scale_from_projection() -> None:
    projection = MapProjection("PROJ", 0.1, -0.1)
    assert projection.name == "PROJ"
    assert projection.xstep == 0.1
    assert projection.ystep == -0.1

    scale = projection.scale
    assert scale.xstep == 0.1
    assert scale.ystep == -0.1

@pytest.mark.parametrize(
    "lhs,rhs,is_equal",
    [
        (MapProjection("A", 0.1, -0.1), MapProjection("A", 0.1, -0.1), True),
        (MapProjection("A", 0.1, -0.1), MapProjection("B", 0.1, -0.1), False),
        (MapProjection("A", 0.1, -0.1), MapProjection("A", 0.1, 0.1), False),
        (MapProjection("A", 0.1, -0.1), MapProjection("A", -0.1, 0.1), False),
        (MapProjection("A", 0.1, -0.1), MapProjection("A", 0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2), -0.1), True),
        (MapProjection("A", 0.1, -0.1), MapProjection("A", 0.1 - (MINIMAL_DEGREE_OF_INTEREST / 2), -0.1), True),
        (MapProjection("A", 0.1, -0.1), MapProjection("A", 0.1, -0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2)), True),
        (MapProjection("A", 0.1, -0.1), MapProjection("A", 0.1, -0.1 - (MINIMAL_DEGREE_OF_INTEREST / 2)), True),
    ]
)
def test_projection_equality(lhs: MapProjection, rhs : MapProjection, is_equal: bool) -> None:
    assert MINIMAL_DEGREE_OF_INTEREST > 0.0
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)
