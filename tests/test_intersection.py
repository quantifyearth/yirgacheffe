import pytest

from helpers import gdal_dataset_of_region
from yirgacheffe.layers import Area, Layer, ConstantLayer, Window


def test_find_intersection_empty_list() -> None:
    with pytest.raises(ValueError):
        Layer.find_intersection([])

def test_find_intersection_single_item() -> None:
    layer = Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    intersection = Layer.find_intersection([layer])
    assert intersection == layer.area

def test_find_intersection_same() -> None:
    layers = [
        Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    ]
    intersection = Layer.find_intersection(layers)
    assert intersection == layers[0].area

def test_find_intersection_subset() -> None:
    layers = [
        Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(gdal_dataset_of_region(Area(-1, 1, 1, -1), 0.02))
    ]
    intersection = Layer.find_intersection(layers)
    assert intersection == layers[1].area

def test_find_intersection_overlap() -> None:
    layers = [
        Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.02))
    ]
    intersection = Layer.find_intersection(layers)
    assert intersection == Area(-10, 10, -5, -5)

def test_find_intersection_distinct() -> None:
    layers = [
        Layer(gdal_dataset_of_region(Area(-110, 10, -100, -10), 0.02)),
        Layer(gdal_dataset_of_region(Area(100, 10, 110, -10), 0.02))
    ]
    with pytest.raises(ValueError):
        _ = Layer.find_intersection(layers)

def test_find_intersection_with_constant() -> None:
    layers = [
        Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        ConstantLayer(1.0)
    ]
    intersection = Layer.find_intersection(layers)
    assert intersection == layers[0].area

def test_find_intersection_different_pixel_pitch() -> None:
    layers = [
        Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.01))
    ]
    with pytest.raises(ValueError):
        _ = Layer.find_intersection(layers)

def test_set_intersection_subset() -> None:
    layer = Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert layer.window == Window(0, 0, 1000, 1000)
    origin_before_pixel = layer.read_array(0, 0, 1, 1)

    intersection = Area(-1.0, 1.0, 1.0, -1.0)

    layer.set_window_for_intersection(intersection)
    assert layer.window == Window(450, 450, 100, 100)
    origin_after_pixel = layer.read_array(0, 0, 1, 1)

    # The default data is populated as a mod of row value, so given
    # were not a multiple of 256 off, these pixels should not have the same
    # value in them
    assert origin_before_pixel[0][0] != origin_after_pixel[0][0]

def test_set_intersection_distinct() -> None:
    layer = Layer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    intersection = Area(-101.0, 1.0, -100.0, -1.0)
    with pytest.raises(ValueError):
        layer.set_window_for_intersection(intersection)
