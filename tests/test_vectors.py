import os
import tempfile

import pytest
from osgeo import gdal

from helpers import make_vectors_with_mutlile_ids, make_vectors_with_id
from yirgacheffe import WSG_84_PROJECTION
from yirgacheffe.layers import RasterLayer, RasteredVectorLayer, VectorLayer, VectorRangeLayer, DynamicVectorRangeLayer
from yirgacheffe.window import Area, PixelScale, Window

def test_basic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            _ = RasteredVectorLayer.layer_from_file(path, "id_no = 123", PixelScale(1.0, -1.0), "WGS 84")

def test_basic_dyanamic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = VectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)
        assert layer.projection == WSG_84_PROJECTION

        del layer

def test_old_dyanamic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = DynamicVectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)
        assert layer.projection == WSG_84_PROJECTION

        del layer

def test_rastered_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = RasteredVectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)
        assert layer.projection == WSG_84_PROJECTION

        del layer

def test_old_rastered_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        layer = VectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)
        assert layer.projection == WSG_84_PROJECTION

        del layer

def test_basic_dynamic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            _ = VectorLayer.layer_from_file(path, "id_no = 123", PixelScale(1.0, -1.0), WSG_84_PROJECTION)

def test_multi_area_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            Area(-10.0, 10.0, 0.0, 0.0),
            Area(0.0, 0.0, 10, -10)
        }
        make_vectors_with_id(42, areas, path)

        rastered_layer = RasteredVectorLayer.layer_from_file(path, "id_no = 42",
            PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        dynamic_layer = VectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)

        for layer in (dynamic_layer, rastered_layer):
            assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
            assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 20)

        window = rastered_layer.window
        for yoffset in range(window.ysize):
            vector_raster = rastered_layer.read_array(0, yoffset, window.xsize, 1)
            dynamic_raster = dynamic_layer.read_array(0, yoffset, window.xsize, 1)
            assert vector_raster.shape == (1, window.xsize)
            assert (vector_raster == dynamic_raster).all()

        del rastered_layer
        del dynamic_layer

def test_empty_layer_from_vector():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        source = VectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)

        empty = RasterLayer.empty_raster_layer_like(source)
        assert empty.pixel_scale == source.pixel_scale
        assert empty.projection == source.projection
        assert empty.window == source.window

@pytest.mark.parametrize(
    "klass",
    [
        VectorLayer,
        RasteredVectorLayer
    ]
)
def test_vector_layers_with_default_burn_value(klass) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WSG_84_PROJECTION)

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        assert total == (layer.window.xsize * layer.window.ysize) / 2

@pytest.mark.parametrize(
    "klass",
    [
        VectorLayer,
        RasteredVectorLayer
    ]
)
def test_vector_layers_with_fixed_burn_value(klass) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WSG_84_PROJECTION, burn_value=5)

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        assert total == ((layer.window.xsize * layer.window.ysize) / 2) * 5

@pytest.mark.parametrize(
    "klass",
    [
        VectorLayer,
        RasteredVectorLayer
    ]
)
def test_vector_layers_with_default_burn_value_and_filter(klass) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        layer = klass.layer_from_file(path, "id_no=42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)

        assert layer.area == Area(-10.0, 10.0, 0.0, 0.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 10, 10)

        # Because we picked one later, all pixels should be burned
        total = layer.sum()
        assert total == (layer.window.xsize * layer.window.ysize)

@pytest.mark.parametrize(
    "klass",
    [
        VectorLayer,
        RasteredVectorLayer
    ]
)
def test_vector_layers_with_field_value(klass) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WSG_84_PROJECTION, burn_value="id_no")

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        assert total == (((layer.window.xsize * layer.window.ysize) / 4) * 42) + \
            (((layer.window.xsize * layer.window.ysize) / 4) * 43)

@pytest.mark.parametrize(
    "value,datatype",
    [
        (1, gdal.GDT_Byte),
        (42, gdal.GDT_Byte),
        (1, gdal.GDT_Int16),
        (42, gdal.GDT_Int16),
        (1024, gdal.GDT_Int16),
        (1.0, gdal.GDT_Float32),
        (0.5, gdal.GDT_Float32),
        (1.0, gdal.GDT_Float64),
    ]
)
def test_vector_layers_with_different_type_burn_value(value, datatype) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 10.0, -10.0), value),
        }
        make_vectors_with_mutlile_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path,
            None,
            PixelScale(1.0, -1.0),
            WSG_84_PROJECTION,
            datatype=datatype,
            burn_value="id_no"
        )

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        assert total == (layer.window.xsize * layer.window.ysize) * value