import numpy as np
import pytest

from helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer, ConstantLayer
from yirgacheffe.operators import LayerOperation

def test_add_byte_layers() -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 + layer2
    comp.save(result)

    expected = data1 + data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

@pytest.mark.parametrize("skip,expected_steps", [
    (1, [0.0, 0.5, 1.0]),
    (2, [0.0, 1.0]),
])
def test_add_byte_layers_with_callback(skip, expected_steps) -> None:
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    callback_possitions = []

    comp = layer1 + layer2
    comp.ystep = skip
    comp.save(result, callback=lambda x: callback_possitions.append(x))
    
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

    expected = data1 * data2
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_add_to_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 + 2.5
    comp.save(result)

    expected = data1 + 2.5
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

    expected = data1 / 2.5
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()

def test_power_float_layer_by_const() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 ** 2.5
    comp.save(result)

    expected = data1 ** 2.5
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

def test_add_to_float_layer_by_np_array() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    result = RasterLayer.empty_raster_layer_like(layer1)

    comp = layer1 + np.array(2.5)
    comp.save(result)

    expected = data1 + 2.5
    actual = result.read_array(0, 0, 4, 2)

    assert (expected == actual).all()
