import numpy

from tests.helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer

def test_add_byte_layers_with_union() -> None:
    data1 = numpy.array([[1, 2, 3, 4,], [5, 6, 7, 8,], [9, 10, 11, 12,], [13, 14, 15, 16,]])
    data2 = numpy.array([[10, 20,], [50, 60,]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((1.0, -1.0), 1.0, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    layers = [layer1, layer2, result]
    window = RasterLayer.find_union(layers)
    for layer in layers:
        layer.set_window_for_union(window)

    comp = layer1 + layer2
    comp.save(result)

    expected = numpy.array([[1, 2, 3, 4,], [5, 16, 27, 8,], [9, 60, 71, 12,], [13, 14, 15, 16,]])
    result.reset_window()
    actual = result.read_array(0, 0, 4, 4)

    assert (expected == actual).all()

def test_add_byte_layers_with_intersection_with_max_save_raster() -> None:
    data1 = numpy.array([[1, 2, 3, 4,], [5, 6, 7, 8,], [9, 10, 11, 12,], [13, 14, 15, 16,]])
    data2 = numpy.array([[10, 20,], [50, 60,]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((1.0, -1.0), 1.0, data2))
    result = RasterLayer.empty_raster_layer_like(layer1)

    layers = [layer1, layer2, result]
    window = RasterLayer.find_intersection(layers)
    for layer in layers:
        layer.set_window_for_intersection(window)

    comp = layer1 + layer2
    comp.save(result)

    expected = numpy.array([[0, 0, 0, 0,], [0, 16, 27, 0,], [0, 60, 71, 0,], [0, 0, 0, 0,]])
    result.reset_window()
    actual = result.read_array(0, 0, 4, 4)

    assert (expected == actual).all()

def test_add_byte_layers_with_intersection_with_min_save_raster() -> None:
    data1 = numpy.array([[1, 2, 3, 4,], [5, 6, 7, 8,], [9, 10, 11, 12,], [13, 14, 15, 16,]])
    data2 = numpy.array([[10, 20,], [50, 60,]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))
    layer2 = RasterLayer(gdal_dataset_with_data((1.0, -1.0), 1.0, data2))
    result = RasterLayer.empty_raster_layer_like(layer2)
    layers = [layer1, layer2, result]

    window = RasterLayer.find_intersection(layers)
    for layer in layers:
        layer.set_window_for_intersection(window)

    comp = layer1 + layer2
    comp.save(result)

    expected = numpy.array([[16, 27,], [60, 71,],])
    result.reset_window()
    actual = result.read_array(0, 0, 2, 2)

    assert (expected == actual).all()
