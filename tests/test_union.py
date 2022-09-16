import pytest

from helpers import make_dataset_of_region
from yirgacheffe.layers import Area, Layer, NullLayer, Window


def test_find_union_empty_list() -> None:
    with pytest.raises(ValueError):
        Layer.find_union([])

def test_find_union_single_item() -> None:
    layer = Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    union = Layer.find_union([layer])
    assert union == layer.area

def test_find_union_same() -> None:
    layers = [
        Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    ]
    union = Layer.find_union(layers)
    assert union == layers[0].area

def test_find_union_subset() -> None:
    layers = [
        Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(make_dataset_of_region(Area(-1, 1, 1, -1), 0.02))
    ]
    union = Layer.find_union(layers)
    assert union == layers[0].area

def test_find_union_overlap() -> None:
    layers = [
        Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(make_dataset_of_region(Area(-15, 15, -5, -5), 0.02))
    ]
    union = Layer.find_union(layers)
    assert union == Area(-15, 15, 10, -10)

def test_find_union_distinct() -> None:
    layers = [
        Layer(make_dataset_of_region(Area(-110, 10, -100, -10), 0.02)),
        Layer(make_dataset_of_region(Area(100, 10, 110, -10), 0.02))
    ]
    union = Layer.find_union(layers)
    assert union == Area(-110, 10, 110, -10)

def test_find_union_with_null() -> None:
    layers = [
        Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        NullLayer()
    ]
    union = Layer.find_union(layers)
    assert union == layers[1].area

def test_find_union_different_pixel_pitch() -> None:
    layers = [
        Layer(make_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        Layer(make_dataset_of_region(Area(-15, 15, -5, -5), 0.01))
    ]
    with pytest.raises(ValueError):
        _ = Layer.find_union(layers)

@pytest.mark.parametrize(
    "left_padding,right_padding,top_padding,bottom_padding",
    [
        (1, 1, 0, 0),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0),
        (1, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 1, 0),
        (0, 0, 1, 0),
        (1, 1, 0, 1),
        (0, 1, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 1, 1, 1),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (0, 0, 1, 1),
    ]
)
def test_set_union_superset(left_padding: int, right_padding: int, top_padding: int, bottom_padding: int) -> None:

    pixel_density = 0.02
    origin_area = Area(-1, 1, 1, -1)

    layer = Layer(make_dataset_of_region(origin_area, pixel_density))
    assert layer.window == Window(0, 0, 100, 100)

    # The make_dataset... function fills rows with the yoffset, and so the first row
    # will be 0s, matching our padding value, so we use the second row here
    origin_before_pixel = layer.read_array(0, 1, 100, 1)
    assert list(origin_before_pixel[0]) == ([1,] * 100)

    # Superset only extends on both sides
    superset = Area(-1 - left_padding, 1 + top_padding, 1 + right_padding, -1 - bottom_padding)
    layer.set_window_for_union(superset)
    assert layer.window == Window(
        (0 - left_padding) / pixel_density,
        (0 - top_padding) / pixel_density,
        (2 + left_padding + right_padding) / pixel_density,
        (2 + top_padding + bottom_padding) / pixel_density,
    )

    origin_after_pixel = layer.read_array(0, 1 + int(top_padding / pixel_density), 100 + int((left_padding + right_padding) / pixel_density), 1)
    assert list(origin_after_pixel[0]) == ([0,] * int(left_padding / pixel_density)) + list(origin_before_pixel[0]) + ([0,] * int(right_padding / pixel_density))
