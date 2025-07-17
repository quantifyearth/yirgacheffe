
import numpy as np
import yirgacheffe
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
                m.setattr(yirgacheffe.constants, "YSTEP", 1)
                c.parallel_save(result)

        expected = np.full((20, 20), 42.0)
        actual = result.read_array(0, 0, 20, 20)

        assert (expected == actual).all()
