
import numpy as np
import pytest

import yirgacheffe as yg
from yirgacheffe.layers import RasterLayer, ConstantLayer
from yirgacheffe.operators import DataType
from yirgacheffe.window import Area, PixelScale

def test_constant_save() -> None:
    area = Area(left=-1.0, right=1.0, top=1.0, bottom=-1.0)
    scale = PixelScale(0.1, -0.1)
    with RasterLayer.empty_raster_layer(area, scale, DataType.Float32) as result:
        with ConstantLayer(42.0) as c:
            c.save(result)

        expected = np.full((20, 20), 42.0)
        actual = result.read_array(0, 0, 20, 20)

        assert (expected == actual).all()

def test_constant_parallel_save(monkeypatch) -> None:
    area = Area(left=-1.0, right=1.0, top=1.0, bottom=-1.0)
    scale = PixelScale(0.1, -0.1)
    with RasterLayer.empty_raster_layer(area, scale, DataType.Float32) as result:
        with ConstantLayer(42.0) as c:
            with monkeypatch.context() as m:
                m.setattr(yg.constants, "YSTEP", 1)
                c.parallel_save(result)

        expected = np.full((20, 20), 42.0)
        actual = result.read_array(0, 0, 20, 20)

        assert (expected == actual).all()

@pytest.mark.parametrize("lhs,rhs,expected_equal", [
    (1, 2, False),
    (1, 1, True),
    (1.0, 2.0, False),
    (1.0, 1.0, True),
    (1, 1.0, True),
])
def test_equality_and_hash(lhs,rhs,expected_equal) -> None:
    a = yg.constant(lhs)
    b = yg.constant(rhs)

    are_equal = a == b
    are_hashed_same = hash(a) == hash(b)

    assert expected_equal == are_equal
    assert expected_equal == are_hashed_same
