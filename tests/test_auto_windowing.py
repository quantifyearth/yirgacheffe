import os
import tempfile

import numpy as np
import pytest

import yirgacheffe as yg
from tests.helpers import gdal_dataset_with_data, make_vectors_with_mutlile_ids
from yirgacheffe.layers import ConstantLayer, RasterLayer, VectorLayer
from yirgacheffe.window import Area

def test_add_windows() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    assert layer1.area != layer2.area
    assert layer1.window != layer2.window

    calc = layer1 + layer2

    assert calc.area == layer2.area
    assert calc.window == layer2.window

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)

    expected = np.array([[11, 22, 30, 40], [53, 64, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])
    actual = result.read_array(0, 0, 4, 4)
    assert (expected == actual).all()

def test_multiply_windows() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    assert layer1.area != layer2.area
    assert layer1.window != layer2.window

    calc = layer1 * layer2

    assert calc.area == layer1.area
    assert calc.window == layer1.window

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)

    expected = np.array([[10, 40], [150, 240]])
    actual = result.read_array(0, 0, 2, 2)
    assert (expected == actual).all()

def test_add_windows_offset() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    layer1 = RasterLayer(gdal_dataset_with_data((-0.02, 0.02), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((-0.04, 0.04), 0.02, data2))

    assert layer1.area != layer2.area
    assert layer1.window != layer2.window

    calc = layer1 + layer2

    assert calc.area == layer2.area

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)

    expected = np.array([[10, 20, 30, 40], [50, 61, 72, 80], [90, 103, 114, 120], [130, 140, 150, 160]])
    actual = result.read_array(0, 0, 4, 4)
    assert (expected == actual).all()

def test_multiply_windows_offset() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    layer1 = RasterLayer(gdal_dataset_with_data((-0.02, 0.02), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((-0.04, 0.04), 0.02, data2))

    assert layer1.area != layer2.area
    assert layer1.window != layer2.window

    calc = layer1 * layer2

    assert calc.area == layer1.area

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)

    expected = np.array([[60, 140], [300, 440]])
    actual = result.read_array(0, 0, 2, 2)
    assert (expected == actual).all()

def test_add_windows_sum() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    layer1 = RasterLayer(gdal_dataset_with_data((-0.02, 0.02), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((-0.04, 0.04), 0.02, data2))

    calc = layer1 + layer2
    total = calc.sum()

    expected = np.array([[10, 20, 30, 40], [50, 61, 72, 80], [90, 103, 114, 120], [130, 140, 150, 160]])
    assert total == np.sum(expected)

def test_multiply_windows_sum() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    layer1 = RasterLayer(gdal_dataset_with_data((-0.02, 0.02), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((-0.04, 0.04), 0.02, data2))

    calc = layer1 * layer2
    total = calc.sum()

    expected = np.array([[60, 140], [300, 440]])
    assert total == np.sum(expected)

def test_constant_layer_result_rhs_add() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(1.0)

    calc = layer1 + const_layer

    assert calc.area == layer1.area

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)
    actual = result.read_array(0, 0, 4, 2)

    expected = 1.0 + data1

    assert (expected == actual).all()

def test_constant_layer_result_lhs_add() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(1.0)
    result = RasterLayer.empty_raster_layer_like(layer1)

    intersection = RasterLayer.find_intersection([layer1, const_layer])
    const_layer.set_window_for_intersection(intersection)
    layer1.set_window_for_intersection(intersection)

    calc = const_layer + layer1

    assert calc.area == layer1.area

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)
    actual = result.read_array(0, 0, 4, 2)

    expected = 1.0 + data1

    assert (expected == actual).all()

def test_constant_layer_result_rhs_multiply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(2.0)

    calc = layer1 * const_layer

    assert calc.area == layer1.area

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)
    actual = result.read_array(0, 0, 4, 2)

    expected = data1 * 2.0

    assert (expected == actual).all()

def test_constant_layer_result_lhs_multiply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(2.0)
    result = RasterLayer.empty_raster_layer_like(layer1)

    intersection = RasterLayer.find_intersection([layer1, const_layer])
    const_layer.set_window_for_intersection(intersection)
    layer1.set_window_for_intersection(intersection)

    calc = const_layer * layer1

    assert calc.area == layer1.area

    result = RasterLayer.empty_raster_layer_like(calc)
    calc.save(result)
    actual = result.read_array(0, 0, 4, 2)

    expected = 2.0 * data1

    assert (expected == actual).all()

def test_vector_layers_add() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.1, data1)) as raster_layer:
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "test.gpkg")
            areas = {
                (Area(-10.0, 10.0, 0.0, 0.0), 42),
                (Area(0.0, 0.0, 10, -10), 43)
            }
            make_vectors_with_mutlile_ids(areas, path)

            burn_value = 2
            with VectorLayer.layer_from_file(
                path,
                None,
                raster_layer.map_projection.scale,
                raster_layer.map_projection.name,
                burn_value=burn_value
            ) as vector_layer:
                layer2_total = vector_layer.sum()
                assert layer2_total == ((vector_layer.window.xsize * vector_layer.window.ysize) / 2) * burn_value

                calc = raster_layer + vector_layer

                assert calc.area == vector_layer.area

                total = calc.sum()
                assert total == layer2_total + np.sum(data1)

                with RasterLayer.empty_raster_layer_like(calc) as result:
                    calc.save(result)
                    total = result.sum()
                    assert total == layer2_total + np.sum(data1)

def test_vector_layers_add_unbound_rhs() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.1, data1)) as raster_layer:
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "test.gpkg")
            areas = {
                (Area(-10.0, 10.0, 0.0, 0.0), 42),
                (Area(0.0, 0.0, 10, -10), 43)
            }
            make_vectors_with_mutlile_ids(areas, path)

            burn_value = 2
            with VectorLayer.layer_from_file(path, None, None, None, burn_value=burn_value) as vector_layer:
                calc = raster_layer + vector_layer

                layer2_total = ((calc.window.xsize * calc.window.ysize) / 2) * burn_value

                assert calc.area != vector_layer.area

                total = calc.sum()
                assert total == layer2_total + np.sum(data1)

                with RasterLayer.empty_raster_layer_like(calc) as result:
                    calc.save(result)
                    total = result.sum()
                    assert total == layer2_total + np.sum(data1)

def test_vector_layers_add_unbound_lhs() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.1, data1)) as raster_layer:
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "test.gpkg")
            areas = {
                (Area(-10.0, 10.0, 0.0, 0.0), 42),
                (Area(0.0, 0.0, 10, -10), 43)
            }
            make_vectors_with_mutlile_ids(areas, path)

            burn_value = 2
            with VectorLayer.layer_from_file(path, None, None, None, burn_value=burn_value) as vector_layer:
                calc = vector_layer + raster_layer

                layer2_total = ((calc.window.xsize * calc.window.ysize) / 2) * burn_value

                assert calc.area != vector_layer.area

                total = calc.sum()
                assert total == layer2_total + np.sum(data1)

                with RasterLayer.empty_raster_layer_like(calc) as result:
                    calc.save(result)
                    total = result.sum()
                    assert total == layer2_total + np.sum(data1)

def test_vector_layers_multiply() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    layer1 = RasterLayer(gdal_dataset_with_data((-1.0, 1.0), 1.0, data1))

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        burn_value = 2
        layer2 = VectorLayer.layer_from_file(path, None, layer1.pixel_scale, layer1.projection, burn_value=burn_value)
        layer2_total = layer2.sum()
        assert layer2_total == ((layer2.window.xsize * layer2.window.ysize) / 2) * burn_value

        calc = layer1 * layer2

        assert calc.area == layer1.area

        result = RasterLayer.empty_raster_layer_like(calc)
        calc.save(result)
        actual = result.read_array(0, 0, 2, 2)

        expected = np.array([[2, 0], [0, 8]])
        assert (expected == actual).all()

@pytest.mark.skipif(yg._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_save_windows() -> None:
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])

    with tempfile.TemporaryDirectory() as tempdir:
        layer1_filename = os.path.join(tempdir, "layer1.tif")
        layer1_dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=layer1_filename)
        layer1_dataset.Close()

        layer2_filename = os.path.join(tempdir, "layer2.tif")
        layer2_dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data2, filename=layer2_filename)
        layer2_dataset.Close()

        layer1 = RasterLayer.layer_from_file(layer1_filename)
        layer2 = RasterLayer.layer_from_file(layer2_filename)

        assert layer1.area != layer2.area
        assert layer1.window != layer2.window

        calc = layer1 + layer2

        assert calc.area == layer2.area
        assert calc.window == layer2.window

        result = RasterLayer.empty_raster_layer_like(calc)
        calc.parallel_save(result)

        expected = np.array([[11, 22, 30, 40], [53, 64, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]])
        actual = result.read_array(0, 0, 4, 4)
        assert (expected == actual).all()
