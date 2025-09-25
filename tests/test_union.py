import tempfile
from pathlib import Path

import pytest

from tests.helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe.window import Area, Window
from yirgacheffe.layers import ConstantLayer, RasterLayer, VectorLayer


def test_find_union_empty_list() -> None:
    with pytest.raises(ValueError):
        _ = RasterLayer.find_union([])

def test_find_union_single_item() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    union = RasterLayer.find_union([layer])
    assert union == layer.area

def test_find_union_same() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    ]
    union = RasterLayer.find_union(layers)
    assert union == layers[0].area

def test_find_union_subset() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-1, 1, 1, -1), 0.02))
    ]
    union = RasterLayer.find_union(layers)
    assert union == layers[0].area

def test_find_union_overlap() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.02))
    ]
    union = RasterLayer.find_union(layers)
    assert union == Area(-15, 15, 10, -10)

def test_find_union_distinct() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-110, 10, -100, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(100, 10, 110, -10), 0.02))
    ]
    union = RasterLayer.find_union(layers)
    assert union == Area(-110, 10, 110, -10)

    for layer in layers:
        layer.set_window_for_union(union)

def test_find_union_with_null() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        ConstantLayer(0.0)
    ]
    union = RasterLayer.find_union(layers)
    assert union == layers[0].area

def test_find_union_different_pixel_pitch() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.01))
    ]
    with pytest.raises(ValueError):
        _ = RasterLayer.find_union(layers)

def test_find_union_with_vector_unbound() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=58, top=74, right=180, bottom=42)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        raster = RasterLayer(gdal_dataset_of_region(Area(left=59.93, top=70.07, right=170.04, bottom=44.98), 0.13))
        vector = VectorLayer.layer_from_file(path, None, None, None)
        assert vector.area == area

        layers = [raster, vector]
        union = RasterLayer.find_union(layers)
        assert union == vector.area

        raster.set_window_for_union(union)
        with pytest.raises(ValueError):
            vector.set_window_for_union(union)


def test_find_union_with_vector_bound() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=58, top=74, right=180, bottom=42)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        raster = RasterLayer(gdal_dataset_of_region(Area(left=59.93, top=70.07, right=170.04, bottom=44.98), 0.13))
        vector = VectorLayer.layer_from_file(path, None, raster.map_projection.scale, raster.map_projection.name)
        assert vector.area != area

        layers = [raster, vector]
        union = RasterLayer.find_union(layers)
        assert union == vector.area

        for layer in layers:
            layer.set_window_for_union(union)

@pytest.mark.parametrize("scale", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
def test_set_union_self(scale) -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), scale))
    old_window = layer.window

    # note that the area we passed to gdal_dataset_of_region isn't pixel aligned, so we must
    # use the area from loading the dataset
    layer.set_window_for_union(layer.area)
    assert layer.window == old_window

    # reset should not do much here
    layer.reset_window()
    assert layer.window == old_window

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

    layer = RasterLayer(gdal_dataset_of_region(origin_area, pixel_density))
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

    origin_after_pixel = layer.read_array(
        0,
        1 + int(top_padding / pixel_density),
        100 + int((left_padding + right_padding) / pixel_density),
        1
    )
    assert list(origin_after_pixel[0]) == (
        [0,] * int(left_padding / pixel_density)) +\
        list(origin_before_pixel[0]) +\
        ([0,] * int(right_padding / pixel_density)
    )

    layer.reset_window()
    assert layer.window == Window(0, 0, 100, 100)
