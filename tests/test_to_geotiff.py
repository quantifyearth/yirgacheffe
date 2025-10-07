import os
import tempfile

import numpy as np
import pytest

import yirgacheffe as yg
from yirgacheffe import DataType
from yirgacheffe.layers import RasterLayer
from yirgacheffe._operators import LayerOperation

from tests.helpers import gdal_dataset_with_data

def test_to_geotiff_on_int_layer() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    assert layer1.datatype == DataType.Int64

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        layer1.to_geotiff(filename)

        with RasterLayer.layer_from_file(filename) as result:
            assert result.datatype == DataType.Int64
            actual = result.read_array(0, 0, 4, 2)
            assert (data1 == actual).all()

def test_to_geotiff_on_float_layer() -> None:
    data1 = np.array([[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    assert layer1.datatype == DataType.Float64

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        layer1.to_geotiff(filename)

        with RasterLayer.layer_from_file(filename) as result:
            assert result.datatype == DataType.Float64
            actual = result.read_array(0, 0, 4, 2)
            assert np.isclose(data1, actual).all()

def test_to_geotiff_single_thread() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    calc = layer1 * 2

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        calc.to_geotiff(filename)

        with RasterLayer.layer_from_file(filename) as result:
            expected = data1 * 2
            actual = result.read_array(0, 0, 4, 2)
            assert (expected == actual).all()

def test_to_geotiff_single_thread_and_sum() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    calc = layer1 * 2

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        actual_sum = calc.to_geotiff(filename, and_sum=True)

        assert (data1.sum() * 2) == actual_sum

        with RasterLayer.layer_from_file(filename) as result:
            expected = data1 * 2
            actual = result.read_array(0, 0, 4, 2)
            assert (expected == actual).all()

@pytest.mark.skipif(yg._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
@pytest.mark.parametrize("parallelism", [
    2,
    True,
])
def test_to_geotiff_parallel_thread(monkeypatch, parallelism) -> None:
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            src_filename = os.path.join("src.tif")
            dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=src_filename)
            dataset.Close()
            with yg.read_raster(src_filename) as layer1:
                calc = layer1 * 2
                filename = os.path.join(tempdir, "test.tif")
                calc.to_geotiff(filename, parallelism=parallelism)

                with RasterLayer.layer_from_file(filename) as result:
                    expected = data1 * 2
                    actual = result.read_array(0, 0, 4, 2)
                    assert (expected == actual).all()

@pytest.mark.skipif(yg._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
@pytest.mark.parametrize("parallelism", [
    2,
    True,
])
def test_to_geotiff_parallel_thread_and_sum(monkeypatch, parallelism) -> None:
    with monkeypatch.context() as m:
        m.setattr(yg.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            src_filename = os.path.join("src.tif")
            dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=src_filename)
            dataset.Close()
            with yg.read_raster(src_filename) as layer1:
                filename = os.path.join(tempdir, "test.tif")
                calc = layer1 * 2
                actual_sum = calc.to_geotiff(filename, and_sum=True, parallelism=parallelism)

                assert (data1.sum() * 2) == actual_sum

                with RasterLayer.layer_from_file(filename) as result:
                    expected = data1 * 2
                    actual = result.read_array(0, 0, 4, 2)
                    assert (expected == actual).all()

def test_to_multiband_geotiff_no_data() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        with pytest.raises(ValueError):
            yg.to_multiband_geotiff(filename, {})

def test_to_multiband_geotiff_on_aligned_layer() -> None:
    data1 = np.array([[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))

    data2 = np.array([[10.1, 20.1, 30.1, 40.1], [50.1, 60.1, 70.1, 80.1]])
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.0, data2))

    data = [data1, data2]
    layers = [layer1, layer2]
    assert layer1.area == layer2.area

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        yg.to_multiband_geotiff(filename, {"layer1": layer1, "layer2": layer2})

        for band in range(2):
            with RasterLayer.layer_from_file(filename, band=band+1) as result:
                assert result._dataset.GetRasterBand(band + 1).GetDescription() == f"layer{band + 1}"
                assert result.area == layers[band].area
                assert result.window == layers[band].window
                assert result.datatype == DataType.Float64
                actual = result.read_array(0, 0, 4, 2)
                assert np.isclose(data[band], actual).all()

def test_to_multiband_geotiff_on_unaligned_layer() -> None:
    data1 = np.array([[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))

    data2 = np.array([[10.1, 20.1, 30.1, 40.1], [50.1, 60.1, 70.1, 80.1]])
    layer2 = RasterLayer(gdal_dataset_with_data((10.0, 10.0), 1.0, data2))

    data = [data1, data2]
    layers = [layer1, layer2]
    union_area = RasterLayer.find_union(layers)

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        yg.to_multiband_geotiff(filename, {"layer1": layer1, "layer2": layer2})

        for band in range(2):
            with RasterLayer.layer_from_file(filename, band=band+1) as result:
                assert result._dataset.GetRasterBand(band + 1).GetDescription() == f"layer{band + 1}"
                assert result.area == union_area

                assert result.datatype == DataType.Float64
                actual = result.read_array(0, 0, 4, 2)
                assert np.isclose(data[band], actual).all()

