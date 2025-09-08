import os
import resource
import tempfile

import numpy as np
import pytest
import torch

import yirgacheffe
from tests.helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer
from yirgacheffe._operators import LayerOperation

# These tests are marked skip for MLX, because there seems to be a problem with
# calling mx.eval in the tests for parallel save on Linux (which is what we use
# for github actions for instance). They do pass on macOS - but that could also
# just be down to Python versions. To add further complication, if I just run
# this file along on linux with MLX it passes fine 🤦
#
# It seems that under the hood MLX is doing some threading of its own and my
# guess is that that's interacting with the Python threading here.

def test_add_byte_layers_with_one_thread_uses_regular_save(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 4)
        m.setattr(LayerOperation, "save", None)
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
            with pytest.raises(TypeError):
                comp.parallel_save(result)

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_add_byte_layers(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
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

            assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_rlimit_nofiles(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "test1.tif")
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
            dataset1.Close()

            rlimit_log = []
            def callback_rlimit_recorder(_progress: float) -> None:
                rlimit_log.append(resource.getrlimit(resource.RLIMIT_NOFILE))

            with RasterLayer.layer_from_file(path1) as layer:
                with RasterLayer.empty_raster_layer_like(layer) as result:

                    before_current_fd_limit, before_max_fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

                    comp = layer * 2
                    comp.parallel_save(result, callback=callback_rlimit_recorder)

                    after_current_fd_limit, after_max_fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

                    assert after_current_fd_limit == before_current_fd_limit
                    assert after_max_fd_limit == before_max_fd_limit

                    assert len(rlimit_log) > 0
                    for recorded_current_fd_limit, recorded_max_fd_limit in rlimit_log:
                        assert recorded_current_fd_limit == before_max_fd_limit
                        assert recorded_max_fd_limit == before_max_fd_limit

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_add_byte_layers_and_sum(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
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
            sum_total = comp.parallel_save(result, and_sum=True)

            expected = data1 + data2
            actual = result.read_array(0, 0, 4, 2)

            assert (expected == actual).all()
            assert sum_total == expected.sum()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_sum(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
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

            comp = layer1 + layer2
            sum_total = comp.parallel_sum()

            expected = data1 + data2
            assert sum_total == expected.sum()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
@pytest.mark.parametrize("skip,expected_steps", [
    (1, [0.0, 0.25, 0.5, 0.75, 1.0]),
    (2, [0.0, 0.5, 1.0]),
])
def test_parallel_with_different_skip(monkeypatch, skip, expected_steps) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "test1.tif")
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]])
            dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
            dataset1.Close()

            path2 = os.path.join(tempdir, "test2.tif")
            data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [10, 20, 30, 40], [50, 60, 70, 80]])
            dataset2 = gdal_dataset_with_data((0.0, 0.0), 0.02, data2, filename=path2)
            dataset2.Close()

            layer1 = RasterLayer.layer_from_file(path1)
            layer2 = RasterLayer.layer_from_file(path2)
            result = RasterLayer.empty_raster_layer_like(layer1)

            callback_possitions = []

            comp = layer1 + layer2
            comp.ystep = skip
            comp.parallel_save(result, callback=callback_possitions.append)

            expected = data1 + data2
            actual = result.read_array(0, 0, 4, 4)

            assert (expected == actual).all()

            assert callback_possitions == expected_steps

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_equality(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "test1.tif")
            data1 = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
            dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
            dataset1.Close()
            with RasterLayer.layer_from_file(path1) as layer1:
                with RasterLayer.empty_raster_layer_like(layer1) as result:
                    comp = layer1 == 2
                    comp.parallel_save(result)

                    expected = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
                    actual = result.read_array(0, 0, 4, 2)

                    assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_equality_to_file(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "test1.tif")
            path2 = os.path.join(tempdir, "result.tif")
            data1 = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
            dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
            dataset1.Close()
            with RasterLayer.layer_from_file(path1) as layer1:
                comp = layer1 == 2
                with RasterLayer.empty_raster_layer_like(layer1, filename=path2) as result:
                    comp.parallel_save(result)
            with RasterLayer.layer_from_file(path2) as actual_result:
                expected = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
                actual = actual_result.read_array(0, 0, 4, 2)
                assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_unary_numpy_apply_with_function(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
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
            comp.ystep = 1
            comp.parallel_save(result)

            expected = data1 + 1.0
            actual = result.read_array(0, 0, 4, 2)

            assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_unary_numpy_apply_with_lambda(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "test1.tif")
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
            dataset1.Close()
            layer1 = RasterLayer.layer_from_file(path1)

            result = RasterLayer.empty_raster_layer_like(layer1)

            comp = layer1.numpy_apply(lambda a: a + 1.0)
            comp.ystep = 1
            comp.parallel_save(result)

            expected = data1 + 1.0
            actual = result.read_array(0, 0, 4, 2)

            assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_where_simple(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "test1.tif")
            data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]])
            dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
            dataset1.Close()
            layer1 = RasterLayer.layer_from_file(path1)

            result = RasterLayer.empty_raster_layer_like(layer1)

            comp = LayerOperation.where(layer1 > 0, 1, 2)
            comp.ystep = 1
            comp.parallel_save(result)

            expected = np.where(data1 > 0, 1, 2)
            actual = result.read_array(0, 0, 4, 2)
            assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_conv2d() -> None:
    with tempfile.TemporaryDirectory() as tempdir:

        data1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]).astype(np.float32)
        weights = np.array([
            [0.0, 0.1, 0.0],
            [0.1, 0.6, 0.1],
            [0.0, 0.1, 0.0],
        ])

        conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        conv.weight = torch.nn.Parameter(torch.from_numpy(np.array([[weights.astype(np.float32)]])))
        tensorres = conv(torch.from_numpy(np.array([[data1]])))
        expected = tensorres.detach().numpy()[0][0]

        path1 = os.path.join(tempdir, "test1.tif")
        dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
        dataset1.Close()

        with RasterLayer.layer_from_file(path1) as layer1:

            calc = layer1.conv2d(weights)
            with RasterLayer.empty_raster_layer_like(layer1) as res:
                calc.save(res)
                actual = res.read_array(0, 0, 5, 5)

                # Torch and MLX give slightly different rounding
                assert np.isclose(expected, actual).all()
