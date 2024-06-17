import os
import tempfile

import numpy as np
import pytest

from helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer, ConstantLayer
from yirgacheffe.operators import LayerOperation

def test_add_byte_layers() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path1 = os.path.join(tempdir, "test1.tif")
        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
        dataset1.Close()

        path2 = os.path.join(tempdir, "test2.tif")
        data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        dataset2 = gdal_dataset_with_data((0.0, 0.0), 0.02, data2, filename=path2)
        dataset2.Close()

        layer1 = RasterLayer.layer_from_file(path1)
        layer2 = RasterLayer.layer_from_file(path2)
        result = RasterLayer.empty_raster_layer_like(layer1)

        comp = layer1 + layer2
        comp.parallel_save(result)

        expected = data1 + data2
        actual = result.read_array(0, 0, 4, 2)

        print(f"expected: {expected}")
        print(f"actual: {actual}")

        assert (expected == actual).all()

@pytest.mark.parametrize("skip,expected_steps", [
    (1, [0.0, 0.5, 1.0]),
    (2, [0.0, 1.0]),
])
def test_parallel_with_different_skip(skip, expected_steps) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path1 = os.path.join(tempdir, "test1.tif")
        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
        dataset1.Close()

        path2 = os.path.join(tempdir, "test2.tif")
        data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        dataset2 = gdal_dataset_with_data((0.0, 0.0), 0.02, data2, filename=path2)
        dataset2.Close()

        layer1 = RasterLayer.layer_from_file(path1)
        layer2 = RasterLayer.layer_from_file(path2)
        result = RasterLayer.empty_raster_layer_like(layer1)

        comp = layer1 + layer2
        comp.ystep = skip
        comp.parallel_save(result)

        expected = data1 + data2
        actual = result.read_array(0, 0, 4, 2)

        assert (expected == actual).all()

def test_parallel_unary_numpy_apply_with_function() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path1 = os.path.join(tempdir, "test1.tif")
        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
        dataset1.Close()
        layer1 = RasterLayer.layer_from_file(path1)

        result = RasterLayer.empty_raster_layer_like(layer1)

        def simple_add(chunk):
            return chunk + 1.0

        comp = layer1.numpy_apply(simple_add)
        comp.parallel_save(result)

        expected = data1 + 1.0
        actual = result.read_array(0, 0, 4, 2)

        assert (expected == actual).all()


def test_parallel_unary_numpy_apply_with_lambda() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path1 = os.path.join(tempdir, "test1.tif")
        data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
        dataset1.Close()
        layer1 = RasterLayer.layer_from_file(path1)

        result = RasterLayer.empty_raster_layer_like(layer1)

        comp = layer1.numpy_apply(lambda a: a + 1.0)
        comp.parallel_save(result)

        expected = data1 + 1.0
        actual = result.read_array(0, 0, 4, 2)

        assert (expected == actual).all()