import tempfile
from pathlib import Path

import numpy as np

import yirgacheffe as yg

def test_save_and_read_simple_geotiff() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test.tif"
        projection = yg.MapProjection("esri:54009", 100.0, -100.0)

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with yg.from_array(data, (0, 0), projection) as layer:
            assert layer.datatype == yg.DataType.Int64
            layer.to_geotiff(filename)

        with yg.read_raster(filename) as layer:
            assert layer.area == yg.Area(0, 0, 400.0, -200.0, projection)
            assert layer.window == yg.Window(0, 0, 4, 2)
            assert layer.nodata is None
            assert layer.datatype == yg.DataType.Int64

            result = layer.read_array(0, 0, 4, 2)
            assert (result == data).all()

def test_save_and_read_nodata_geotiff() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test.tif"
        projection = yg.MapProjection("esri:54009", 100.0, -100.0)

        data = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        with yg.from_array(data, (0, 0), projection) as layer:
            assert layer.datatype == yg.DataType.Float64
            layer.to_geotiff(filename, nodata=3)

        with yg.read_raster(filename) as layer:
            assert layer.area == yg.Area(0, 0, 400.0, -200.0, projection)
            assert layer.window == yg.Window(0, 0, 4, 2)
            assert layer.nodata == 3.0
            assert layer.datatype == yg.DataType.Float64

            result = layer.read_array(0, 0, 4, 2)
            data[data==3] = np.nan
            assert np.allclose(result, data, equal_nan=True)
