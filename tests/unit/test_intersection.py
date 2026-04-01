import tempfile
from pathlib import Path

import pytest
from osgeo import gdal

import yirgacheffe as yg
from tests.unit.helpers import (
    gdal_dataset_of_region,
    gdal_empty_dataset_of_region,
    make_vectors_with_id,
)
from yirgacheffe import Area, MapProjection
from yirgacheffe._layers import RasterLayer, ConstantLayer, H3CellLayer, VectorLayer
from yirgacheffe._datatypes import Window


def test_find_intersection_empty_list() -> None:
    with pytest.raises(ValueError):
        _ = yg.find_intersection([])


def test_find_intersection_single_item() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    intersection = yg.find_intersection([layer])
    assert intersection == layer.area


def test_find_intersection_same() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
    ]
    intersection = yg.find_intersection(layers)
    assert intersection == layers[0].area


def test_find_intersection_subset() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-1, 1, 1, -1), 0.02)),
    ]
    intersection = yg.find_intersection(layers)
    assert intersection == layers[1].area


def test_find_intersection_overlap() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.02)),
    ]
    intersection = yg.find_intersection(layers)
    assert intersection == Area(-10, 10, -5, -5, layers[0].projection)


def test_find_intersection_distinct() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-110, 10, -100, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(100, 10, 110, -10), 0.02)),
    ]
    with pytest.raises(ValueError):
        _ = yg.find_intersection(layers)


def test_find_intersection_with_constant() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        ConstantLayer(1.0),
    ]
    intersection = yg.find_intersection(layers)
    assert intersection == layers[0].area

    for layer in layers:
        updated_layer = layer.as_area(intersection)
        assert updated_layer.area == intersection


def test_find_intersection_with_vector_unbound() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=58, top=74, right=180, bottom=42)
        make_vectors_with_id(42, {area}, path)
        assert path.exists()

        raster = RasterLayer(
            gdal_dataset_of_region(
                Area(left=-180.05, top=90.09, right=180.05, bottom=-90.09), 0.13
            )
        )
        vector = VectorLayer.layer_from_file(path, None, None, None)
        assert vector.area == area

        layers = [raster, vector]
        intersection = yg.find_intersection(layers)

        expected_area = area.project_like(raster.area)
        assert intersection == expected_area

        _ = raster.as_area(intersection)
        with pytest.raises(ValueError):
            _ = vector.as_area(intersection)


def test_find_intersection_with_vector_bound() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=58, top=74, right=180, bottom=42)
        make_vectors_with_id(42, {area}, path)
        assert path.exists()

        raster = RasterLayer(
            gdal_dataset_of_region(
                Area(left=-180.05, top=90.09, right=180.05, bottom=-90.09), 0.13
            )
        )
        vector = VectorLayer.layer_from_file_like(path, raster)
        assert vector.area != area

        layers = [raster, vector]
        intersection = yg.find_intersection(layers)
        assert intersection == vector.area

        for layer in layers:
            _ = layer.as_area(intersection)


def test_find_intersection_with_vector_awkward_rounding() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "test.gpkg"
        area = Area(left=-90, top=45, right=90, bottom=-45)
        make_vectors_with_id(42, {area}, path)
        assert path.exists()

        raster = RasterLayer(
            gdal_dataset_of_region(Area(left=-180, top=90, right=180, bottom=-90), 18.0)
        )
        vector = VectorLayer.layer_from_file_like(path, raster)

        rounded_area = Area(
            left=-90,
            top=54,
            right=90,
            bottom=-54,
            projection=MapProjection("epsg:4326", 18.0, -18.0),
        )
        assert vector.area == rounded_area

        layers = [raster, vector]
        intersection = yg.find_intersection(layers)
        assert intersection == vector.area

        for layer in layers:
            _ = layer.as_area(intersection)


def test_find_intersection_different_pixel_pitch() -> None:
    layers = [
        RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02)),
        RasterLayer(gdal_dataset_of_region(Area(-15, 15, -5, -5), 0.01)),
    ]
    with pytest.raises(ValueError):
        _ = yg.find_intersection(layers)


@pytest.mark.parametrize(
    "scale", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
)
def test_set_intersection_self(scale) -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), scale))
    old_dimensions = layer.dimensions
    old_window = layer._virtual_window

    # note that the area we passed to gdal_dataset_of_region isn't pixel aligned, so we must
    # use the area from loading the dataset
    clipped_layer = layer.as_area(layer.area)
    assert clipped_layer.dimensions == layer.dimensions


def test_set_intersection_subset() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert layer.dimensions == (1000, 1000)
    assert layer._virtual_window == Window(0, 0, 1000, 1000)
    origin_before_pixel = layer.read_array(0, 0, 1, 1)

    intersection = Area(-1.0, 1.0, 1.0, -1.0)

    clipped_layer = layer.as_area(intersection)
    assert clipped_layer.dimensions == (100, 100)
    origin_after_pixel = clipped_layer.read_array(0, 0, 1, 1)

    # The default data is populated as a mod of row value, so given
    # were not a multiple of 256 off, these pixels should not have the same
    # value in them
    assert origin_before_pixel[0][0] != origin_after_pixel[0][0]


def test_set_intersection_distinct() -> None:
    layer = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    intersection = Area(-101.0, 1.0, -100.0, -1.0)
    out_of_area_layer = layer.as_area(intersection)
    xsize, ysize = out_of_area_layer.dimensions
    data = out_of_area_layer.read_array(0, 0, xsize, ysize)
    assert data.sum() == 0


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
        RasterLayer(
            gdal_empty_dataset_of_region(
                Area(
                    left=-180.00082337073326,
                    top=90.00041168536663,
                    right=180.00082337073326,
                    bottom=-90.00041168536663,
                ),
                0.0008983152841195215,
            )
        ),
        RasterLayer(
            gdal_empty_dataset_of_region(
                Area(
                    left=-180.00082337073326,
                    top=90.00041168536661,
                    right=180.00082337073326,
                    bottom=-90.00041168536664,
                ),
                0.0008983152841195216,
            )
        ),
        RasterLayer(
            gdal_empty_dataset_of_region(
                Area(
                    left=-180,
                    top=90.00041168536661,
                    right=180,
                    bottom=-90.00041168536664,
                ),
                0.0008983152841195215,
            )
        ),
        RasterLayer(
            gdal_empty_dataset_of_region(
                Area(
                    left=-3.6372785853999425,
                    top=47.767016917771436,
                    right=3.578888091932174,
                    bottom=42.068104755317194,
                ),
                0.0008983152841195215,
            )
        ),
    ]

    intersection = yg.find_intersection(layers)
    assert intersection == layers[-1].area
    clipped_layers = [layer.as_area(intersection) for layer in layers]
    for other in clipped_layers[1:]:
        assert clipped_layers[0].dimensions == other.dimensions


def test_intersection_stability() -> None:
    # This test uses h3 tiles as a lazy way to get some bounded regions,
    # but the bug this test exercises is not h3 specific. This was another case of
    # a rounding error that causes set_window_for_* methods to wobble depending on how far
    # away from the top left thing where. adding round_down_pixels fixed this.
    cells = [
        "874b93aaeffffff",
        "874b93a85ffffff",
        "874b93aa3ffffff",
        "874b93a84ffffff",
        "874b93a80ffffff",
    ]
    projection = MapProjection("epsg:4326", 0.000898315284120, -0.000898315284120)

    tiles = [H3CellLayer(cell_id, projection) for cell_id in cells]

    # composing the same tiles within different areas should not cause them to
    # wobble around
    union = yg.find_union(tiles)
    superunion = union.grow(25 * projection.xstep)

    scratch1 = RasterLayer.empty_raster_layer(
        union, projection.scale, gdal.GDT_Float64, name="s1"
    )
    scratch2 = RasterLayer.empty_raster_layer(
        superunion, projection.scale, gdal.GDT_Float64, name="s2"
    )

    relative_offsets = {}

    for scratch in [scratch1, scratch2]:
        offsets = []
        first = None
        for tile in tiles:
            layers = [scratch, tile]
            intersection = yg.find_intersection(layers)

            # We know the tile is a subset of the scratch region, so
            # the intersection should just be that
            assert intersection == tile.area

            clipped_layers = [layer.as_area(intersection) for layer in layers]

            if first is None:
                first = scratch._virtual_window
            else:
                offset = (
                    scratch._virtual_window.xoff - first.xoff,
                    scratch._virtual_window.yoff - first.yoff,
                )
                offsets.append(offset)
        relative_offsets[scratch.name] = offsets

    assert relative_offsets[scratch1.name] == relative_offsets[scratch2.name]
