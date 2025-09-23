import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import yirgacheffe
from yirgacheffe.window import Area, PixelScale
from yirgacheffe.layers import ConstantLayer, RasterLayer, VectorLayer
from yirgacheffe.operators import DataType
from yirgacheffe._operators import LayerOperation
from yirgacheffe._backends import backend
from tests.helpers import gdal_dataset_with_data, gdal_dataset_of_region, make_vectors_with_id

def test_add_byte_layers() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.uint8)
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]]).astype(np.uint8)

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    assert layer1.datatype == DataType.Byte
    assert layer2.datatype == DataType.Byte

    comp = layer1 + layer2
    comp.save(result)

    expected = data1 + data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_error_of_pixel_scale_wrong() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.uint8)
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]]).astype(np.uint8)
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.01, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

    with pytest.raises(ValueError):
        _ = layer1 + layer2

def test_error_of_pixel_scale_wrong_three_param() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.uint8)
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]]).astype(np.uint8)
    data3 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]]).astype(np.uint8)
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    layer3 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.01, data3))

    with pytest.raises(ValueError):
        _ = LayerOperation.where(layer1, layer2, layer3)

def test_incompatible_source_and_destination_projections() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    with RasterLayer.empty_raster_layer(layer1.area, PixelScale(1.0, -1.0), layer1.datatype) as result:
        with pytest.raises(ValueError):
            layer1.save(result)

@pytest.mark.parametrize("skip,expected_steps", [
    (1, [0.0, 0.5, 1.0]),
    (2, [0.0, 1.0]),
])
def test_add_byte_layers_with_callback(skip, expected_steps) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.uint8)
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]]).astype(np.uint8)

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    assert layer1.datatype == DataType.Byte
    assert layer2.datatype == DataType.Byte

    callback_possitions = []

    comp = layer1 + layer2
    comp.ystep = skip
    comp.save(result, callback=callback_possitions.append)

    expected = data1 + data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

    assert callback_possitions == expected_steps

def test_sub_byte_layers() -> None:
    data1 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    data2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 - layer2
    comp.save(result)

    expected = data1 - data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_add_float_layers() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 + layer2
    comp.save(result)

    expected = data1 + data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_sub_float_layers() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 - layer2
    comp.save(result)

    expected = data1 - data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_mult_float_layers() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 * layer2
    comp.save(result)

    expected = backend.promote(data1) * backend.promote(data2)
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

def test_div_float_layers() -> None:
    data1 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    data2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.5, 6.5, 7.5, 8.5]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 / layer2
    comp.save(result)

    expected = backend.promote(data1) / backend.promote(data2)
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

def test_floor_div_float_layers() -> None:
    data1 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    data2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.5, 6.5, 7.5, 8.5]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 // layer2
    comp.save(result)

    expected = backend.promote(data1) // backend.promote(data2)
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

def test_remainder_float_layers() -> None:
    data1 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    data2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.5, 6.5, 7.5, 8.5]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 % layer2
    comp.save(result)

    expected = backend.promote(data1) % backend.promote(data2)
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

@pytest.mark.parametrize("c", [
    (float(2.5)),
    (int(2)),
    (np.uint16(2)),
    (np.float32(2.5)),
])
def test_add_to_float_layer_by_const(c) -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 + c
    comp.save(result)

    expected = data1 + c
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_sub_from_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 - 0.5
    comp.save(result)

    expected = data1 - 0.5
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_mult_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 * 2.5
    comp.save(result)

    expected = data1 * 2.5
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_div_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 / 2.5
    comp.save(result)

    expected = backend.promote(data1) / 2.5
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

def test_floordiv_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 // 2.5
    comp.save(result)

    expected = backend.promote(data1) // 2.5
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

def test_remainder_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 % 2.5
    comp.save(result)

    expected = backend.promote(data1) % 2.5
    backend.eval_op(expected)

    actual = backend.demote_array(result.read_array(0, 0, 4, 2))

    assert (expected == actual).all()

def test_power_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 ** 2.5
    comp.save(result)

    expected = backend.promote(data1) ** 2.5
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_simple_unary_numpy_apply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(chunk):
        return chunk + 1.0

    comp = layer1.numpy_apply(simple_add)
    comp.save(result)

    expected = data1 + 1.0
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_isin_unary_numpy_apply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(chunk):
        return np.isin(chunk, [2.0, 3.0])

    comp = layer1.numpy_apply(simple_add)
    comp.save(result)

    # The * 1.0 is because the numpy result will be bool, but we bounced
    # our answer via a float gdal dataset
    expected = np.isin(data1, [2.0, 3.0]) * 1.0
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_simple_binary_numpy_apply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(chunk1, chunk2):
        return chunk1 + chunk2

    comp = layer1.numpy_apply(simple_add, layer2)
    comp.save(result)

    expected = data1 + data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_simple_unary_shader_apply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(pixel):
        return pixel + 1.0

    comp = layer1.shader_apply(simple_add)
    comp.save(result)

    expected = data1 + 1.0
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_simple_binary_shader_apply() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(pixel1, pixel2):
        return pixel1 + pixel2

    comp = layer1.shader_apply(simple_add, layer2)
    comp.save(result)

    expected = data1 + data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

@pytest.mark.parametrize("operator",
    [
        lambda a, b: a == b,
        lambda a, b: a != b,
        lambda a, b: a > b,
        lambda a, b: a >= b,
        lambda a, b: a < b,
        lambda a, b: a <= b,
    ]
)
def test_comparison_float_layer_by_const(operator) -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = operator(layer1, 3.0)
    comp.save(result)

    expected = operator(data1, 3.0)
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_sum_layer() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    # a no-op just to get us into an operator
    comp = layer1 + 0.0

    actual = comp.sum()

    expected = np.sum(data1)
    assert expected == actual

def test_constant_layer_result_rhs() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(1.0)
    result = RasterLayer.empty_raster_layer_like(layer1)

    layers = [layer1, const_layer, result]
    intersection = RasterLayer.find_intersection(layers)
    for layer in layers:
        layer.set_window_for_intersection(intersection)

    comp = layer1 + const_layer
    comp.save(result)
    result.reset_window()
    actual = result.read_array(0, 0, 4, 2)

    expected = 1.0 + data1

    assert (expected == actual).all()

def test_constant_layer_result_lhs() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(1.0)
    result = RasterLayer.empty_raster_layer_like(layer1)

    intersection = RasterLayer.find_intersection([layer1, const_layer])
    const_layer.set_window_for_intersection(intersection)
    layer1.set_window_for_intersection(intersection)

    comp = const_layer + layer1
    comp.save(result)

    actual = result.read_array(0, 0, 4, 2)

    expected = 1.0 + data1

    assert (expected == actual).all()

def test_shader_apply_constant_lhs() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(1.0)
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(pixel1, pixel2):
        return pixel1 + pixel2

    comp = const_layer.shader_apply(simple_add, layer1)
    comp.save(result)

    expected = data1 + 1.0
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_shader_apply_constant_rhs() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    const_layer = ConstantLayer(1.0)
    result = RasterLayer.empty_raster_layer_like(layer1)

    def simple_add(pixel1, pixel2):
        return pixel1 + pixel2

    comp = layer1.shader_apply(simple_add, const_layer)
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp.save(result)

    expected = data1 + 1.0
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_direct_layer_save() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    layer1.save(result)
    actual = result.read_array(0, 0, 4, 2)

    assert (data1 == actual).all()

def test_direct_layer_sum() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    actual = layer1.sum()

    expected = np.sum(data1)
    assert expected == actual

def test_direct_layer_sum_chunked() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    calc = LayerOperation(layer1)
    calc.ystep = 1
    actual = calc.sum()

    expected = np.sum(data1)
    assert expected == actual

def test_direct_layer_min() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    actual = layer1.min()

    expected = np.min(data1)
    assert expected == actual

def test_direct_layer_min_chunked() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    calc = LayerOperation(layer1)
    calc.ystep = 1
    actual = calc.min()

    expected = np.min(data1)
    assert expected == actual

def test_direct_layer_max() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    actual = layer1.max()

    expected = np.max(data1)
    assert expected == actual

def test_direct_layer_max_chunked() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))

    calc = LayerOperation(layer1)
    calc.ystep = 1
    actual = calc.max()

    expected = np.max(data1)
    assert expected == actual

def test_direct_layer_save_and_sum() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    actual_sum = layer1.save(result, and_sum=True)
    actual_data = result.read_array(0, 0, 4, 2)
    expected_sum = np.sum(data1)

    assert (data1 == actual_data).all()
    assert expected_sum == actual_sum

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_add_to_float_layer_by_np_array() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 + np.array(2.5)
    comp.save(result)

    expected = data1 + 2.5
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_write_mulitband_raster() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    layers = [layer1, layer2]

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test.tif")
        with RasterLayer.empty_raster_layer_like(layer1, filename=filename, bands=2) as result:
            for i in range(2):
                layers[i].save(result, band=i+1)

        for i in range(2):
            layer = RasterLayer.layer_from_file(filename, band=i+1)
            expected = layers[i].read_array(0, 0, 4, 2)
            actual = layer.read_array(0, 0, 4, 2)

            assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_save_and_sum_float32(monkeypatch) -> None:
    random.seed(42)
    data = []
    for _ in range(10):
        row = []
        for _ in range(10):
            row.append(random.random())
        data.append(row)

    data1 = np.array(data, dtype=np.float32)
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    assert layer1.datatype == DataType.Float32

    # Sum forces things to float64
    expected = np.sum(data1.astype(np.float64))

    with monkeypatch.context() as m:
        for blocksize in range(1,11):
            m.setattr(yirgacheffe.constants, "YSTEP", blocksize)
            with RasterLayer.empty_raster_layer_like(layer1) as store:
                actual = layer1.save(store, and_sum=True)
            assert expected == actual

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_parallel_save_and_sum_float32(monkeypatch) -> None:
    random.seed(42)
    data = []
    for _ in range(10):
        row = []
        for _ in range(10):
            row.append(random.random())
        data.append(row)

    with tempfile.TemporaryDirectory() as tempdir:
        path1 = os.path.join(tempdir, "test1.tif")
        data1 = np.array(data, dtype=np.float32)
        dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=path1)
        dataset1.Close()
        layer1 = RasterLayer.layer_from_file(path1)
        assert layer1.datatype == DataType.Float32

        # Sum forces things to float64
        expected = np.sum(data1.astype(np.float64))

        with monkeypatch.context() as m:
            for blocksize in range(1,11):
                m.setattr(yirgacheffe.constants, "YSTEP", blocksize)
                with RasterLayer.empty_raster_layer_like(layer1) as store:
                    actual = layer1.parallel_save(store, and_sum=True)
                assert expected == actual

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
def test_sum_float32(monkeypatch) -> None:
    random.seed(42)
    data = []
    for _ in range(10):
        row = []
        for _ in range(10):
            row.append(random.random())
        data.append(row)

    data1 = np.array(data, dtype=np.float32)
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    assert layer1.datatype == DataType.Float32

    # Sum forces things to float64
    expected = np.sum(data1.astype(np.float64))

    with monkeypatch.context() as m:
        for blocksize in range(1,11):
            m.setattr(yirgacheffe.constants, "YSTEP", blocksize)
            actual = layer1.sum()
            assert expected == actual

def test_and_byte_layers() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]]).astype(np.uint8)
    data2 = np.array([[1, 1, 0, 0], [2, 2, 2, 2]]).astype(np.uint8)

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    assert layer1.datatype == DataType.Byte
    assert layer2.datatype == DataType.Byte

    comp = layer1 & layer2
    comp.save(result)

    expected = data1 & data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_or_byte_layers() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]]).astype(np.uint8)
    data2 = np.array([[1, 1, 0, 0], [2, 2, 2, 2]]).astype(np.uint8)

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    assert layer1.datatype == DataType.Byte
    assert layer2.datatype == DataType.Byte

    comp = layer1 | layer2
    comp.save(result)

    expected = data1 | data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_and_int_layers() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]]).astype(np.int16)
    data2 = np.array([[1, 1, 0, 0], [2, 2, 2, 2]]).astype(np.int16)

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    assert layer1.datatype == DataType.Int16
    assert layer2.datatype == DataType.Int16

    comp = layer1 & layer2
    comp.save(result)

    expected = data1 & data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_or_int_layers() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]]).astype(np.int16)
    data2 = np.array([[1, 1, 0, 0], [2, 2, 2, 2]]).astype(np.int16)

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    assert layer1.datatype == DataType.Int16
    assert layer2.datatype == DataType.Int16

    comp = layer1 | layer2
    comp.save(result)

    expected = data1 | data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_nan_to_num() -> None:
    data1 = np.array([[float('nan'), float('nan'), float('nan'), float('nan')], [1, 2, 3, 0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.nan_to_num(nan=42)
    comp.ystep = 1
    comp.save(result)

    expected = np.array([[42, 42, 42, 42], [1, 2, 3, 0]])
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_nan_to_num_default() -> None:
    data1 = np.array([[float('nan'), float('nan'), float('nan'), float('nan')], [1, 2, 3, 0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.nan_to_num()
    comp.ystep = 1
    comp.save(result)

    expected = np.array([[0, 0, 0, 0], [1, 2, 3, 0]])
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

@pytest.mark.parametrize("ct", [
    float,
    int,
    np.uint16,
    np.float32,
])
def test_where_simple(ct) -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.where(layer1 > 0, ct(1), ct(2))
    comp.ystep = 1
    comp.save(result)

    expected = np.where(data1 > 0, ct(1), ct(2))
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_where_layers() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]])
    data_a = np.array([[10, 10, 10, 10], [20, 20, 20, 20]])
    data_b = np.array([[100, 100, 100, 100], [200, 200, 200, 200]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer_a = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data_a))
    layer_b = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data_b))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.where(layer1 > 0, layer_a, layer_b)
    comp.ystep = 1
    comp.save(result)

    expected = np.where(data1 > 0, data_a, data_b)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_isin_simple_method() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.isin([2, 3])
    comp.ystep = 1
    comp.save(result)

    expected = backend.isin(backend.promote(data1), [2, 3])


    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_isin_simple_module() -> None:
    data1 = np.array([[0, 1, 0, 2], [0, 0, 1, 1]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.isin(layer1, [2, 3])
    comp.ystep = 1
    comp.save(result)

    expected = np.isin(data1, [2, 3])
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

@pytest.mark.parametrize("val", [
    (float(2.0)),
    (int(2)),
    (np.uint16(2)),
    (np.float32(2.0)),
])
def test_layer_comparison_to_value(val) -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 == val
    comp.save(result)

    expected = data1 == val
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_less_than_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 < 2.0
    comp.save(result)

    expected = data1 < 2.0
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_less_than_or_equal_to_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 <= 2.0
    comp.save(result)

    expected = data1 <= 2.0
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_greater_than_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 > 2.0
    comp.save(result)

    expected = data1 > 2.0
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_greater_than_or_equal_to_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 >= 2.0
    comp.save(result)

    expected = data1 >= 2.0
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_comparison_to_layer() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [7.0, 2.0, 5.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 == layer2
    comp.save(result)

    expected = data1 == data2
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_less_than_layer() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [7.0, 2.0, 5.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 < layer2
    comp.save(result)

    expected = data1 < data2
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_less_than_or_equal_to_layer() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [7.0, 2.0, 5.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 <= layer2
    comp.save(result)

    expected = data1 <= data2
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_greater_than_layer() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [7.0, 2.0, 5.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 > layer2
    comp.save(result)

    expected = data1 > data2
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_layer_greater_than_or_equal_to_layer() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [7.0, 2.0, 5.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 >= layer2
    comp.save(result)

    expected = data1 >= data2
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_log_method() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.log()
    comp.save(result)

    expected = backend.log(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_log_module() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.log(layer1)
    comp.save(result)

    expected = backend.log(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_log2() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.log2()
    comp.save(result)

    expected = backend.log2(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_log10() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.log10()
    comp.save(result)

    expected = backend.log10(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_exp_method() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.exp()
    comp.save(result)

    expected = backend.exp(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_exp_module() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.exp(layer1)
    comp.save(result)

    expected = backend.exp(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_exp2() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.exp2()
    comp.save(result)

    expected = backend.exp2(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_minimum_layers() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [8.0, 2.0, 5.0, 7.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.minimum(layer1, layer2)
    comp.save(result)

    expected = np.minimum(data1, data2)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_maximum_layers() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    data2 = np.array([[3.0, 2.0, 1.0, 4.0], [8.0, 2.0, 5.0, 7.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.maximum(layer1, layer2)
    comp.save(result)

    expected = np.maximum(data1, data2)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_clip_both_method() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.clip(3.0, 6.0)
    comp.save(result)

    expected = data1.clip(3.0, 6.0)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_clip_both_module() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.clip(layer1, 3.0, 6.0)
    comp.save(result)

    expected = np.clip(data1, 3.0, 6.0)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_clip_upper_method() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.clip(max=6.0)
    comp.save(result)

    expected = data1.clip(max=6.0)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_clip_upper_module() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.clip(layer1, max=6.0)
    comp.save(result)

    expected = np.clip(data1, a_min=None, a_max=6.0)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_clip_lower_method() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.clip(min=3.0)
    comp.save(result)

    expected = data1.clip(min=3.0)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_clip_lower_module() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.clip(layer1, min=3.0)
    comp.save(result)

    expected = np.clip(data1, a_min=3.0, a_max=None)
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_abs_method() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.abs()
    comp.save(result)

    expected = backend.abs_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_abs_module() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.abs(layer1)
    comp.save(result)

    expected = backend.abs_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

@pytest.mark.parametrize("skip", [
    1,
    2,
    5,
])
def test_simple_conv2d_unity(skip) -> None:
    data1 = np.array([
        [0, 0, 5, 0, 0],
        [0, 1, 1, 1, 0],
        [4, 1, 1, 1, 3],
        [0, 1, 1, 1, 0],
        [0, 0, 2, 0, 0],
    ]).astype(np.float32)
    weights = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1)) as layer1:
        calc = layer1.conv2d(weights)
        calc.ystep = skip
        with RasterLayer.empty_raster_layer_like(layer1) as res:
            calc.save(res)
            actual = res.read_array(0, 0, 5, 5)
            assert (data1 == actual).all()

@pytest.mark.parametrize("skip", [
    1,
    2,
    5,
])
def test_simple_conv2d_blur(skip) -> None:
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

    with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1)) as layer1:

        calc = layer1.conv2d(weights)
        calc.ystep = skip
        with RasterLayer.empty_raster_layer_like(layer1) as res:
            calc.save(res)
            actual = res.read_array(0, 0, 5, 5)

            # Torch and MLX give slightly different rounding
            assert np.isclose(expected, actual).all()

@pytest.mark.parametrize("skip", [
    1,
    2,
    5,
])
def test_simple_conv2d_over_calculated_result(skip) -> None:
    # This test is interesting as it'll pull expanded data from the child calculation
    # datasets
    data1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]).astype(np.float32)
    data2 = np.array([
        [2, 0, 0, 0, 2],
        [0, 2, 0, 2, 0],
        [0, 0, 2, 0, 0],
        [0, 2, 0, 2, 0],
        [2, 0, 0, 0, 2],
    ]).astype(np.float32)
    weights = np.array([
        [0.0, 0.1, 0.0],
        [0.1, 0.6, 0.1],
        [0.0, 0.1, 0.0],
    ])

    joined_data = data1 * data2

    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    conv.weight = torch.nn.Parameter(torch.from_numpy(np.array([[weights.astype(np.float32)]])))
    tensorres = conv(torch.from_numpy(np.array([[joined_data]])))
    expected = tensorres.detach().numpy()[0][0]

    with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1)) as layer1:
        with RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2)) as layer2:

            calc = (layer1 * layer2).conv2d(weights)
            calc.ystep = skip
            with RasterLayer.empty_raster_layer_like(layer1) as res:
                calc.save(res)
                actual = res.read_array(0, 0, 5, 5)

                # Torch and MLX give slightly different rounding
                assert np.isclose(expected, actual).all()

def test_floor_method() -> None:
    data1 = np.array([[1.0, 1.2, 1.5, 1.7], [-1.0, -1.2, -1.5, -1.7]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.floor()
    comp.save(result)

    expected = backend.floor_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_floor_module() -> None:
    data1 = np.array([[1.0, 1.2, 1.5, 1.7], [-1.0, -1.2, -1.5, -1.7]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.floor(layer1)
    comp.save(result)

    expected = backend.floor_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_round_method() -> None:
    data1 = np.array([[1.0, 1.2, 1.5, 1.7], [-1.0, -1.2, -1.5, -1.7]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.round()
    comp.save(result)

    expected = backend.round_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_round_module() -> None:
    data1 = np.array([[1.0, 1.2, 1.5, 1.7], [-1.0, -1.2, -1.5, -1.7]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.round(layer1)
    comp.save(result)

    expected = backend.round_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_ceil_method() -> None:
    data1 = np.array([[1.0, 1.2, 1.5, 1.7], [-1.0, -1.2, -1.5, -1.7]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1.ceil()
    comp.save(result)

    expected = backend.ceil_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

def test_ceil_module() -> None:
    data1 = np.array([[1.0, 1.2, 1.5, 1.7], [-1.0, -1.2, -1.5, -1.7]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = LayerOperation.ceil(layer1)
    comp.save(result)

    expected = backend.ceil_op(backend.promote(data1))
    backend.eval_op(expected)

    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

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

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
@pytest.mark.parametrize("parallelism", [
    2,
    True,
])
def test_to_geotiff_parallel_thread(monkeypatch, parallelism) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            src_filename = os.path.join("src.tif")
            dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=src_filename)
            dataset.Close()
            with yirgacheffe.read_raster(src_filename) as layer1:
                calc = layer1 * 2
                filename = os.path.join(tempdir, "test.tif")
                calc.to_geotiff(filename, parallelism=parallelism)

                with RasterLayer.layer_from_file(filename) as result:
                    expected = data1 * 2
                    actual = result.read_array(0, 0, 4, 2)
                    assert (expected == actual).all()

@pytest.mark.skipif(yirgacheffe._backends.BACKEND != "NUMPY", reason="Only applies for numpy")
@pytest.mark.parametrize("parallelism", [
    2,
    True,
])
def test_to_geotiff_parallel_thread_and_sum(monkeypatch, parallelism) -> None:
    with monkeypatch.context() as m:
        m.setattr(yirgacheffe.constants, "YSTEP", 1)
        m.setattr(LayerOperation, "save", None)
        with tempfile.TemporaryDirectory() as tempdir:
            data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            src_filename = os.path.join("src.tif")
            dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1, filename=src_filename)
            dataset.Close()
            with yirgacheffe.read_raster(src_filename) as layer1:
                filename = os.path.join(tempdir, "test.tif")
                calc = layer1 * 2
                actual_sum = calc.to_geotiff(filename, and_sum=True, parallelism=parallelism)

                assert (data1.sum() * 2) == actual_sum

                with RasterLayer.layer_from_file(filename) as result:
                    expected = data1 * 2
                    actual = result.read_array(0, 0, 4, 2)
                    assert (expected == actual).all()

def test_raster_and_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        raster = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 1.0))
        assert raster.sum() > 0.0

        path = Path(tempdir) / "test.gpkg"
        area = Area(-5.0, 5.0, 5.0, -5.0)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        vector = VectorLayer.layer_from_file(path, None, PixelScale(1.0, -1.0), yirgacheffe.WGS_84_PROJECTION)

        calc = raster * vector
        assert calc.sum() > 0.0
        assert calc.sum() < raster.sum()

def test_raster_and_vector_mixed_projection() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        raster = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 1.1))
        assert raster.sum() > 0.0

        path = Path(tempdir) / "test.gpkg"
        area = Area(-5.0, 5.0, 5.0, -5.0)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        vector = VectorLayer.layer_from_file(path, None, PixelScale(1.0, -1.0), yirgacheffe.WGS_84_PROJECTION)

        with pytest.raises(ValueError):
            _ = raster * vector

def test_raster_and_vector_no_scale_on_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        raster = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 1.0))
        assert raster.sum() > 0.0

        path = Path(tempdir) / "test.gpkg"
        area = Area(-5.0, 5.0, 5.0, -5.0)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        vector = VectorLayer.layer_from_file(path, None, None, None)

        calc = raster * vector
        assert calc.sum() > 0.0
        assert calc.sum() < raster.sum()

def test_isnan() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 5.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    dataset.GetRasterBand(1).SetNoDataValue(5.0)
    with RasterLayer(dataset) as layer:
        calc = layer.isnan()
        with RasterLayer.empty_raster_layer_like(calc) as result:
            calc.save(result)
            actual = result.read_array(0, 0, 4, 2)
            expected = data1 == 5.0
            assert (expected == actual).all()
