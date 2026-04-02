import tempfile
from pathlib import Path

import numpy as np
import pytest

import yirgacheffe as yg
from yirgacheffe._layers import RasterLayer


def test_constant_default_behaviour() -> None:
    c = yg.constant(42)
    assert c.projection is None
    assert c.area == yg.Area.world()
    with pytest.raises(AttributeError):
        _ = c.dimensions


def test_constant_save() -> None:
    area = yg.Area(left=-1.0, right=1.0, top=1.0, bottom=-1.0, projection=yg.MapProjection("epsg:4326", 0.1, -0.1))
    with RasterLayer.empty_raster_layer(area, yg.DataType.Float32) as result:
        with yg.constant(42.0) as c:
            c.save(result)

        expected = np.full((20, 20), 42.0)
        actual = result.read_array(0, 0, 20, 20)

        assert (expected == actual).all()


def test_constant_parallel_save(monkeypatch) -> None:
    area = yg.Area(left=-1.0, right=1.0, top=1.0, bottom=-1.0, projection=yg.MapProjection("epsg:4326", 0.1, -0.1))
    with RasterLayer.empty_raster_layer(area, yg.DataType.Float32) as result:
        with monkeypatch.context() as m:
            m.setattr(yg.constants, "YSTEP", 1)
            with yg.constant(42.0) as c:
                c.parallel_save(result)

        expected = np.full((20, 20), 42.0)
        actual = result.read_array(0, 0, 20, 20)

        assert (expected == actual).all()


@pytest.mark.parametrize(
    "lhs,rhs,expected_equal",
    [
        (1, 2, False),
        (1, 1, True),
        (1.0, 2.0, False),
        (1.0, 1.0, True),
        (1, 1.0, True),  # This is Python standard behaviour
    ],
)
def test_cse_hash(lhs, rhs, expected_equal) -> None:
    a = yg.constant(lhs)
    b = yg.constant(rhs)

    assert a is not b
    assert a.name != b.name

    are_hashed_same = a._cse_hash == b._cse_hash
    assert expected_equal == are_hashed_same


@pytest.mark.parametrize("projection", [
    yg.MapProjection("epsg:4326", 0.01, -0.01),
    yg.MapProjection("esri:54009", 100, -100),
])
def test_constant_to_geotiff(projection) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.tif"
        area = yg.Area(0, 0, 100 * projection.xstep, 100 * projection.ystep, projection)
        with yg.constant(42) as const:
            assert const.area == yg.Area.world()
            aread_const = const.as_area(area)
            assert aread_const.area == area
            aread_const.to_geotiff(path)

        with yg.read_raster(path) as result:
            assert result.dimensions == (100, 100)
            assert result.area == area
            data = result.read_array(0, 0, 100, 100)
            assert (data == 42).all()
