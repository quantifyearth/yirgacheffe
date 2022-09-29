import os
import tempfile

import numpy
import pytest

from helpers import gdal_dataset_of_region, make_vectors_with_id

from yirgacheffe.layers import Area, Layer, PixelScale, Window, VectorRangeLayer, DynamicVectorRangeLayer

# There is a lot of "del" in this file, due to a combination of gdal having no way
# to explicitly close a file beyond forcing the gdal object's deletion, and Windows
# getting upset that it tries to clear up the TemporaryDirectory and there's an open
# file handle within that directory.

def test_make_basic_layer() -> None:
    area = Area(-10, 10, 10, -10)
    layer = Layer(gdal_dataset_of_region(area, 0.02))
    assert layer.area == area
    assert layer.pixel_scale == (0.02, -0.02)
    assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
    assert layer.window == Window(0, 0, 1000, 1000)

def test_layer_from_null() -> None:
    # Seems a petty test, but gdal doesn't throw exceptions
    # so you often get None datasets if you're not careful
    with pytest.raises(ValueError):
        Layer(None)

def test_layer_from_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        Layer.layer_from_file("this_file_does_not_exist.tif")

def test_open_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        del dataset
        assert os.path.exists(path)
        layer = Layer.layer_from_file(path)
        assert layer.area == area
        assert layer.pixel_scale == (0.02, -0.02)
        assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
        assert layer.window == Window(0, 0, 1000, 1000)
        del layer

def test_basic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = VectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), "WGS 84")
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)

        del layer

def test_basic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            _ = VectorRangeLayer(path, "id_no = 123", PixelScale(1.0, -1.0), "WGS 84")

def test_basic_dyanamic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = DynamicVectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), "WGS 84")
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)

        del layer

def test_basic_dynamic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            _ = DynamicVectorRangeLayer(path, "id_no = 123", PixelScale(1.0, -1.0), "WGS 84")

def test_multi_area_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            Area(-10.0, 10.0, 0.0, 0.0),
            Area(0.0, 0.0, 10, -10)
        }
        make_vectors_with_id(42, areas, path)

        vector_layer = VectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), "WGS 84")
        dynamic_layer = DynamicVectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), "WGS 84")

        for layer in (dynamic_layer, vector_layer):
            assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
            assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 20)

        window = vector_layer.window
        for yoffset in range(window.ysize):
            print(yoffset)
            vector_raster = vector_layer.read_array(0, yoffset, window.xsize, 1)
            dynamic_raster = dynamic_layer.read_array(0, yoffset, window.xsize, 1)

            assert (vector_raster == dynamic_raster).all()

        del vector_layer
        del dynamic_layer
