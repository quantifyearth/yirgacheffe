import tempfile
from pathlib import Path

import pytest
from osgeo import gdal

from tests.helpers import gdal_dataset_of_region, gdal_empty_dataset_of_region, make_vectors_with_id
from yirgacheffe.window import Area, MapProjection, Window
from yirgacheffe.layers import RasterLayer, ConstantLayer, H3CellLayer, VectorLayer
from yirgacheffe import WGS_84_PROJECTION


def test_find_intersection_empty_list() -> None:
    with pytest.raises(ValueError):
        _ = RasterLayer.find_intersection([])

def test_find_intersection_single_item() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    intersection = RasterLayer.find_intersection([layer])
    assert intersection == layer.area

def test_find_intersection_same() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    ]
    intersection = RasterLayer.find_intersection(layers)
    assert intersection == layers[0].area

def test_find_intersection_subset() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-1, 1, 1, -1), 0.02))
    ]
    intersection = RasterLayer.find_intersection(layers)
    assert intersection == layers[1].area

def test_find_intersection_overlap() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.02))
    ]
    intersection = RasterLayer.find_intersection(layers)
    assert intersection == Area(-10, 10, -5, -5)

def test_find_intersection_distinct() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-110, 10, -100, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(100, 10, 110, -10), 0.02))
    ]
    with pytest.raises(ValueError):
        _ = RasterLayer.find_intersection(layers)

def test_find_intersection_with_constant() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        ConstantLayer(1.0)
    ]
    intersection = RasterLayer.find_intersection(layers)
    assert intersection == layers[0].area

    for layer in layers:
        layer.set_window_for_intersection(intersection)

def test_find_intersection_with_vector_unbound() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=58, top=74, right=180, bottom=42)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        raster = RasterLayer(gdal_dataset_of_region(Area(left=-180.05, top=90.09, right=180.05, bottom=-90.09), 0.13))
        vector = VectorLayer.layer_from_file(path, None, None, None)
        assert vector.area == area

        layers = [raster, vector]
        intersection = RasterLayer.find_intersection(layers)
        assert intersection == vector.area

        raster.set_window_for_intersection(intersection)
        with pytest.raises(ValueError):
            vector.set_window_for_intersection(intersection)

def test_find_intersection_with_vector_bound() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=58, top=74, right=180, bottom=42)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        raster = RasterLayer(gdal_dataset_of_region(Area(left=-180.05, top=90.09, right=180.05, bottom=-90.09), 0.13))
        vector = VectorLayer.layer_from_file(path, None, raster.map_projection.scale, raster.map_projection.name)
        assert vector.area != area

        layers = [raster, vector]
        intersection = RasterLayer.find_intersection(layers)
        assert intersection == vector.area

        for layer in layers:
            layer.set_window_for_intersection(intersection)

def test_find_intersection_with_vector_awkward_rounding() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=-90, top=45, right=90, bottom=-45)
        make_vectors_with_id(42, {area}, path)
        assert path.exists

        raster = RasterLayer(gdal_dataset_of_region(Area(left=-180, top=90, right=180, bottom=-90), 18.0))
        vector = VectorLayer.layer_from_file(path, None, raster.map_projection.scale, raster.map_projection.name)

        rounded_area = Area(left=-90, top=54, right=90, bottom=-54)
        assert vector.area == rounded_area

        layers = [raster, vector]
        intersection = RasterLayer.find_intersection(layers)
        assert intersection == vector.area

        for layer in layers:
            layer.set_window_for_intersection(intersection)

def test_find_intersection_different_pixel_pitch() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.01))
    ]
    with pytest.raises(ValueError):
        _ = RasterLayer.find_intersection(layers)

@pytest.mark.parametrize("scale", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
def test_set_intersection_self(scale) -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), scale))
    old_window = layer.window

    # note that the area we passed to gdal_dataset_of_region isn't pixel aligned, so we must
    # use the area from loading the dataset
    layer.set_window_for_intersection(layer.area)
    assert layer.window == old_window

    # reset should not do much here
    layer.reset_window()
    assert layer.window == old_window

def test_set_intersection_subset() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
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

    layer.reset_window()
    assert layer.window == Window(0, 0, 1000, 1000)

def test_set_intersection_distinct() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    intersection = Area(-101.0, 1.0, -100.0, -1.0)
    with pytest.raises(ValueError):
        layer.set_window_for_intersection(intersection)

def test_find_intersection_nearly_same() -> None:
    # This testcase is based on a real instance we hit whereby
    # the layers were effectively the same, and intended to be the same,
    # but a rounding error of less than the floating point epsilon was in
    # one of the files.
    #
    # gdalinfo rounds the numbers, so it wasn't obvious, but inspecting the
    # GEOTiffs with tifffile (a pure tiff library) showed the error, and GDAL
    # in python showed the error too.
    #
    # Yirgacheffe at the time blew up as it knew to ignore the difference
    # when doing a comparison (thanks to layers.py::almost_equal(a,b)), but
    # when you then multiplied this up by the area it rounded poorly.
    layers = [
        RasterLayer(gdal_empty_dataset_of_region(
            Area(left=-180.00082337073326, top=90.00041168536663, right=180.00082337073326, bottom=-90.00041168536663),
            0.0008983152841195215
        )),
        RasterLayer(gdal_empty_dataset_of_region(
            Area(left=-180.00082337073326, top=90.00041168536661, right=180.00082337073326, bottom=-90.00041168536664),
            0.0008983152841195216
        )),
        RasterLayer(gdal_empty_dataset_of_region(
            Area(left=-180, top=90.00041168536661, right=180, bottom=-90.00041168536664),
            0.0008983152841195215
        )),
        RasterLayer(gdal_empty_dataset_of_region(
            Area(left=-3.6372785853999425, top=47.767016917771436, right=3.578888091932174, bottom=42.068104755317194),
            0.0008983152841195215
        ))
    ]

    intersection = RasterLayer.find_intersection(layers)
    assert intersection == layers[-1].area
    for layer in layers:
        layer.set_window_for_intersection(intersection)
    for other in layers[1:]:
        assert layers[0].window.xsize == other.window.xsize
        assert layers[0].window.ysize == other.window.ysize

def test_intersection_stability() -> None:
    # This test uses h3 tiles as a lazy way to get some bounded regions,
    # but the bug this test exercises is not h3 specific. This was another case of
    # a rounding error that causes set_window_for_* methods to wobble depending on how far
    # away from the top left thing where. adding round_down_pixels fixed this.
    cells = ["874b93aaeffffff", "874b93a85ffffff", "874b93aa3ffffff", "874b93a84ffffff", "874b93a80ffffff"]
    projection = MapProjection(WGS_84_PROJECTION, 0.000898315284120,-0.000898315284120)

    tiles = [
        H3CellLayer(cell_id, projection)
    for cell_id in cells]

    # composing the same tiles within different areas should not cause them to
    # wobble around
    union = RasterLayer.find_union(tiles)
    superunion = union.grow(0.02)

    scratch1 = RasterLayer.empty_raster_layer(union, projection.scale, gdal.GDT_Float64, name='s1')
    scratch2 = RasterLayer.empty_raster_layer(superunion, projection.scale, gdal.GDT_Float64, name='s2')

    relative_offsets = {}

    for scratch in [scratch1, scratch2]:
        offsets = []
        first = None
        for tile in tiles:
            scratch.reset_window()
            layers = [scratch, tile]
            intersection = RasterLayer.find_intersection(layers)

            # We know the tile is a subset of the scratch region, so
            # the intersection should just be that
            assert intersection == tile.area

            for layer in layers:
                layer.set_window_for_intersection(intersection)

            if first is None:
                first = scratch.window
            else:
                offset = scratch.window.xoff - first.xoff, scratch.window.yoff - first.yoff
                offsets.append(offset)
        relative_offsets[scratch.name] = offsets

    assert relative_offsets[scratch1.name] == relative_offsets[scratch2.name]
