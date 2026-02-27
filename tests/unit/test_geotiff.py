import tempfile
from pathlib import Path

import numpy as np
import pytest

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

def test_save_multiband_simple() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test.tif"
        projection = yg.MapProjection("esri:54009", 100.0, -100.0)

        data = [np.full((4, 4), idx) for idx in range(5)]
        layers = [yg.from_array(datum, (0, 0), projection) for datum in data]
        labels = [f"label {idx}" for idx in range(5)]

        yg.to_geotiff(filename, layers, labels)

        for idx in range(5):
            with yg.read_raster(filename, band=idx + 1) as layer:
                assert layer.area == layers[idx].area
                assert layer.name == labels[idx]
                result = layer.read_array(0, 0, 4, 4)
                assert (result == data[idx]).all()

def test_error_mismatched_labels() -> None:
    # 5 layers, 4 labels
    projection = yg.MapProjection("esri:54009", 100.0, -100.0)
    data = [np.full((4, 4), idx) for idx in range(5)]
    layers = [yg.from_array(datum, (0, 0), projection) for datum in data]
    labels = [f"label {idx}" for idx in range(4)]

    with pytest.raises(ValueError):
        yg.to_geotiff("will_not_work.tif", layers, labels)

def test_error_on_mixed_layer_datatypes() -> None:
    projection = yg.MapProjection("esri:54009", 100.0, -100.0)
    layer1 = yg.from_array(
        np.full((4,4), 1).astype(np.int64),
        (0, 0),
        projection,
    )
    layer2 = yg.from_array(
        np.full((4,4), 1.5).astype(np.float32),
        (0, 0),
        projection,
    )
    with pytest.raises(TypeError):
        yg.to_geotiff("will_not_work.tif", [layer1, layer2])

def test_error_on_no_layers() -> None:
    with pytest.raises(ValueError):
        yg.to_geotiff("will_not_work.tif", [])

def test_save_multiband_non_overlapping_area() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test.tif"
        projection = yg.MapProjection("esri:54009", 1.0, -1.0)

        data = [np.full((4, 4), idx) for idx in range(5)]
        layers = [yg.from_array(datum, (idx * 10, idx * 10), projection)
            for idx, datum in enumerate(data)]

        yg.to_geotiff(filename, layers)

        expected_area = yg.Area(
            0.0,
            40.0,
            44.0,
            -4.0,
            projection,
        )
        expected_window = yg.Window(0, 0, 44, 44)

        for idx in range(5):
            with yg.read_raster(filename, band=idx + 1) as layer:
                assert layer.area == expected_area
                assert layer.window == expected_window

def test_error_on_mixed_layer_projection() -> None:
    layer1 = yg.from_array(
        np.full((4,4), 1),
        (0, 0),
        yg.MapProjection("esri:54009", 100.0, -100.0),
    )
    layer2 = yg.from_array(
        np.full((4,4), 1),
        (0, 0),
        yg.MapProjection("epsg:4326", 1.0, -1.0),
    )
    with pytest.raises(ValueError):
        yg.to_geotiff("will_not_work.tif", [layer1, layer2])
