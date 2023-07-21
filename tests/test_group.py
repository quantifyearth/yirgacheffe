import math
import os
import tempfile

import pytest

from helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe import WGS_84_PROJECTION
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

        vector1 = VectorLayer.layer_from_file(path, None, PixelScale(0.2, -0.2), WGS_84_PROJECTION)
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
        vector1 = VectorLayer.layer_from_file(path1, None, PixelScale(0.2, -0.2), WGS_84_PROJECTION)

        path2 = os.path.join(tempdir, "test2.gpkg")
        area2 = Area(-0.0, 10.0, 20.0, -10.0)
        make_vectors_with_id(24, {area2}, path2)
        vector2 = VectorLayer.layer_from_file(path2, None, PixelScale(0.2, -0.2), WGS_84_PROJECTION)

        group = GroupLayer([vector1, vector2])
        assert group.area == Area(-10, 10, 20, -10)
        assert group.window == Window(0, 0, 150, 100)
        assert group.sum() == vector1.sum() + (vector2.sum() / 2)

def test_with_window_adjust():
    with tempfile.TemporaryDirectory() as tempdir:
        layers = []
        for i in range(1, 11):
            path = os.path.join(tempdir, f"{i}.gpkg")
            area = Area(i, 10, i+1, -10)
            make_vectors_with_id(i, {area}, path)
            vector = VectorLayer.layer_from_file(path, None, PixelScale(0.1, -0.1), WGS_84_PROJECTION, burn_value="id_no")
            layers.append(vector)

        group = GroupLayer(layers)
        assert group.area == Area(1, 10, 11, -10)
        assert group.window == Window(0, 0, 100, 200)

        # Test before we apply a window
        row = group.read_array(0, 0, 100, 1)[0]
        for i in range(len(row)):
            assert row[i] == math.ceil((i + 1) / 10.0)

        # Test also for manual offsets that we get expected result
        for i in range(0, 10):
            row = group.read_array(i * 10, 0, 10, 1)[0]
            assert (row == (i + 1)).all()

        # now apply a window over each zone and check we
        # get what we expect
        for i in range(1, 11):
            group.reset_window()
            area = Area(i, 10, i+1, -10)
            group.set_window_for_intersection(area)
            row = group.read_array(0, 0, 10, 1)
            assert (row == i).all()
