import pyproj
import pytest

from yirgacheffe.window import MapProjection
from yirgacheffe.rounding import MINIMAL_DEGREE_OF_INTEREST

@pytest.mark.parametrize("crs", ["epsg:4326", "esri:54009"])
def test_scale_from_projection(crs) -> None:
    name = pyproj.CRS.from_string(crs).to_wkt()
    projection = MapProjection(name, 0.1, -0.1)
    assert projection.name == name
    assert projection.xstep == 0.1
    assert projection.ystep == -0.1

    scale = projection.scale
    assert scale.xstep == 0.1
    assert scale.ystep == -0.1

PROJ_A = pyproj.CRS.from_string("epsg:4326").to_wkt()
PROJ_B = pyproj.CRS.from_string("esri:54009").to_wkt()

@pytest.mark.parametrize(
    "lhs,rhs,is_equal",
    [
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1, -0.1), True),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_B, 0.1, -0.1), False),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1, 0.1), False),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, -0.1, 0.1), False),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2), -0.1), True),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1 - (MINIMAL_DEGREE_OF_INTEREST / 2), -0.1), True),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1, -0.1 + (MINIMAL_DEGREE_OF_INTEREST / 2)), True),
        (MapProjection(PROJ_A, 0.1, -0.1), MapProjection(PROJ_A, 0.1, -0.1 - (MINIMAL_DEGREE_OF_INTEREST / 2)), True),
    ]
)
def test_projection_equality(lhs: MapProjection, rhs : MapProjection, is_equal: bool) -> None:
    assert MINIMAL_DEGREE_OF_INTEREST > 0.0
    assert (lhs == rhs) == is_equal
    assert (lhs != rhs) == (not is_equal)

def test_invalid_projection_name() -> None:
    with pytest.raises(ValueError):
        _ = MapProjection("random name", 1.0, -1.0)
