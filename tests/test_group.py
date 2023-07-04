import os
import tempfile

import pytest

from helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe import WSG_84_PROJECTION
from yirgacheffe.layers import GroupLayer, RasterLayer, VectorLayer
from yirgacheffe.window import Area, PixelScale, Window

def test_empty_group():
    with pytest.raises(ValueError):
        _ = GroupLayer(set())

def test_single_raster_layer_in_group():
    area = Area(-10, 10, 10, -10)
    raster1 = RasterLayer(gdal_dataset_of_region(area, 0.2))
    assert raster1.area == area
    assert raster1.window == Window(0, 0, 100, 100)
    assert raster1.sum() != 0

    group = GroupLayer([raster1])
    assert group.area == area
    assert group.window == Window(0, 0, 100, 100)
    assert group.sum() == raster1.sum()

def test_mismatched_layers():
    area1 = Area(-10, 10, 10, -10)
    raster1 = RasterLayer(gdal_dataset_of_region(area1, 0.2))
    area2 = Area(-10, 10, 10, -10)
    raster2 = RasterLayer(gdal_dataset_of_region(area2, 0.3))

    with pytest.raises(ValueError):
        _ = GroupLayer([raster1, raster2])

def test_two_raster_areas_side_by_side():
    area1 = Area(-10, 10, 10, -10)
    raster1 = RasterLayer(gdal_dataset_of_region(area1, 0.2))
    area2 = Area(10, 10, 30, -10)
    raster2 = RasterLayer(gdal_dataset_of_region(area2, 0.2))

    group = GroupLayer([raster1, raster2])
    assert group.area == Area(-10, 10, 30, -10)
    assert group.window == Window(0, 0, 200, 100)
    assert group.sum() == raster1.sum() + raster2.sum()

def test_two_raster_areas_top_to_bottom():
    area1 = Area(-10, 10, 10, -10)
    raster1 = RasterLayer(gdal_dataset_of_region(area1, 0.2))
    area2 = Area(-10, -10, 10, -30)
    raster2 = RasterLayer(gdal_dataset_of_region(area2, 0.2))

    group = GroupLayer([raster1, raster2])
    assert group.area == Area(-10, 10, 10, -30)
    assert group.window == Window(0, 0, 100, 200)
    assert group.sum() == raster1.sum() + raster2.sum()

def test_single_vector_layer_in_group():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, -10.0)
        make_vectors_with_id(42, {area}, path)

        vector1 = VectorLayer.layer_from_file(path, None, PixelScale(0.2, -0.2), WSG_84_PROJECTION)
        assert vector1.area == area
        assert vector1.window == Window(0, 0, 100, 100)
        assert vector1.sum() == (vector1.window.xsize * vector1.window.ysize)

        group = GroupLayer([vector1])
        assert group.area == area
        assert group.window == Window(0, 0, 100, 100)
        assert group.sum() == vector1.sum()

def test_overlapping_vector_layers():
    with tempfile.TemporaryDirectory() as tempdir:
        path1 = os.path.join(tempdir, "test1.gpkg")
        area1 = Area(-10.0, 10.0, 10.0, -10.0)
        make_vectors_with_id(42, {area1}, path1)
        vector1 = VectorLayer.layer_from_file(path1, None, PixelScale(0.2, -0.2), WSG_84_PROJECTION)

        path2 = os.path.join(tempdir, "test2.gpkg")
        area2 = Area(-0.0, 10.0, 20.0, -10.0)
        make_vectors_with_id(42, {area2}, path2)
        vector2 = VectorLayer.layer_from_file(path2, None, PixelScale(0.2, -0.2), WSG_84_PROJECTION)

        group = GroupLayer([vector1, vector2])
        assert group.area == Area(-10, 10, 20, -10)
        assert group.window == Window(0, 0, 150, 100)
        assert group.sum() == vector1.sum() + (vector2.sum() / 2)
