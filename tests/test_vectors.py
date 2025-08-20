import os
import tempfile

import pytest

from tests.helpers import make_vectors_with_mutlile_ids, make_vectors_with_id, make_vectors_with_empty_feature
from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import RasterLayer, RasteredVectorLayer, VectorLayer
from yirgacheffe.window import Area, PixelScale, Window
from yirgacheffe.operators import DataType

def test_basic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            with RasteredVectorLayer.layer_from_file(path, "id_no = 123", PixelScale(1.0, -1.0), "WGS 84") as _layer:
                pass

def test_basic_dynamic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with VectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WGS_84_PROJECTION) as layer:
            assert layer.area == area
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 10)
            assert layer.projection == WGS_84_PROJECTION
            assert layer.map_projection.name == WGS_84_PROJECTION

            # The astype here is to catch escaping MLX types...
            res = layer.read_array(0, 0, 20, 20).astype(int)
            assert res.sum() > 0

def test_rastered_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with RasteredVectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WGS_84_PROJECTION) as layer:
            assert layer.area == area
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 10)
            assert layer.projection == WGS_84_PROJECTION
            assert layer.map_projection.name == WGS_84_PROJECTION

def test_basic_dynamic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            _ = VectorLayer.layer_from_file(path, "id_no = 123", PixelScale(1.0, -1.0), WGS_84_PROJECTION)

def test_multi_area_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            Area(-10.0, 10.0, 0.0, 0.0),
            Area(0.0, 0.0, 10, -10)
        }
        make_vectors_with_id(42, areas, path)

        rastered_layer = RasteredVectorLayer.layer_from_file(path, "id_no = 42",
            PixelScale(1.0, -1.0), WGS_84_PROJECTION)
        dynamic_layer = VectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WGS_84_PROJECTION)

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
        area = Area(left=44.00253688814017, top=-12.440948032828079, right=50.483612168477286, bottom=-25.1535466075739)
        make_vectors_with_id(42, {area}, path)

        source = VectorLayer.layer_from_file(
            path,
            "id_no = 42",
            PixelScale(xstep=0.00026949458523585647, ystep=-0.00026949458523585647),
            WGS_84_PROJECTION
        )

        empty = RasterLayer.empty_raster_layer_like(source)
        assert empty.pixel_scale == source.pixel_scale
        assert empty.projection == source.projection
        assert empty.map_projection == source.map_projection
        assert empty.window == source.window
        assert empty.area == source.area

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

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WGS_84_PROJECTION)

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)
        assert layer.datatype == DataType.Byte

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

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WGS_84_PROJECTION, burn_value=5)

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

        layer = klass.layer_from_file(path, "id_no=42", PixelScale(1.0, -1.0), WGS_84_PROJECTION)

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
def test_vector_layers_with_invalid_burn_value(klass) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_mutlile_ids(areas, path)

        with pytest.raises(ValueError):
            _ = klass.layer_from_file(
                path,
                None,
                PixelScale(1.0, -1.0),
                WGS_84_PROJECTION,
                burn_value="this_is_wrong"
            )


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

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WGS_84_PROJECTION, burn_value="id_no")

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        assert total == (((layer.window.xsize * layer.window.ysize) / 4) * 42) + \
            (((layer.window.xsize * layer.window.ysize) / 4) * 43)

@pytest.mark.parametrize(
    "value,expected",
    [
        (1, DataType.Byte),
        (42, DataType.Byte),
        (-1, DataType.Int16),
        (1024, DataType.UInt16),
        (1024*1024, DataType.UInt32),
        (-1024*1024, DataType.Int32),
        (1.0, DataType.Float64),
    ]
)
def test_vector_layers_with_guessed_type_burn_value(value, expected) -> None:
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
            WGS_84_PROJECTION,
            burn_value=value
        )

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)
        assert layer.datatype == expected

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        assert total == (layer.window.xsize * layer.window.ysize) * value

@pytest.mark.parametrize(
    "value,datatype",
    [
        (1, DataType.Byte),
        (42, DataType.Byte),
        (1, DataType.Int16),
        (42, DataType.Int16),
        (1024, DataType.Int16),
        (1.0, DataType.Float32),
        (0.5, DataType.Float32),
        (1.0, DataType.Float64),
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
            WGS_84_PROJECTION,
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


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, DataType.Int64),
        (1.0, DataType.Float64),
    ]
)
def test_vector_layers_with_guess_field_type_burn_value(value, expected) -> None:
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
            WGS_84_PROJECTION,
            burn_value="id_no"
        )

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)
        assert layer.datatype == expected

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        assert total == (layer.window.xsize * layer.window.ysize) * value


@pytest.mark.parametrize("size,expect_success",
    [
        ((5, 5), True),
        ((5, 1), True),
        ((1, 5), True),
        ((5, 0), False),
        ((0, 5), False),
    ]
)
def test_read_array_size(size, expect_success):
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        source = RasteredVectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WGS_84_PROJECTION)

        if expect_success:
            data = source.read_array(0, 0, size[0], size[1])
            assert data.shape == (size[1], size[0])
        else:
            with pytest.raises(ValueError):
                _ = source.read_array(0, 0, size[0], size[1])

@pytest.mark.parametrize("anchor,area,expected",
    [
        ((0.0, 0.0), Area(-10.0, 10.0, 10.0, -10.0), Area(-10.0, 10.0, 10.0, -10.0)),
        ((0.0, 0.0), Area(-9.9, 9.9, 9.9, -9.9), Area(-10.0, 10.0, 10.0, -10.0)),
        ((0.0, 0.0), Area(-9.1, 9.1, 9.1, -9.1), Area(-10.0, 10.0, 10.0, -10.0)),

        ((0.0, 0.0), Area(5.0, 10.0, 10.0, 5.0), Area(5.0, 10.0, 10.0, 5.0)),
        ((0.0, 0.0), Area(5.0, -5.0, 10.0, -10.0), Area(5.0, -5.0, 10.0, -10.0)),
        ((0.0, 0.0), Area(-10.0, -5.0, -5.0, -10.0), Area(-10.0, -5.0, -5.0, -10.0)),
        ((0.0, 0.0), Area(-10.0, 10.0, -5.0, 5.0), Area(-10.0, 10.0, -5.0, 5.0)),

        ((0.0, 0.0), Area(5.1, 9.9, 9.9, 5.1), Area(5.0, 10.0, 10.0, 5.0)),
        ((0.0, 0.0), Area(5.1, -5.1, 9.9, -9.9), Area(5.0, -5.0, 10.0, -10.0)),
        ((0.0, 0.0), Area(-9.9, -5.1, -5.1, -9.9), Area(-10.0, -5.0, -5.0, -10.0)),
        ((0.0, 0.0), Area(-9.9, 9.9, -5.1, 5.1), Area(-10.0, 10.0, -5.0, 5.0)),

        ((0.1, 0.1), Area(-10.0, 10.0, 10.0, -10.0), Area(-10.9, 10.1, 10.1, -10.9)),

        ((0.1, 0.1), Area(5.0, 10.0, 10.0, 5.0), Area(4.1, 10.1, 10.1, 4.1)),
        ((0.1, 0.1), Area(5.0, -5.0, 10.0, -10.0), Area(4.1, -4.9, 10.1, -10.9)),
        ((0.1, 0.1), Area(-10.0, -5.0, -5.0, -10.0), Area(-10.9, -4.9, -4.9, -10.9)),
        ((0.1, 0.1), Area(-10.0, 10.0, -5.0, 5.0), Area(-10.9, 10.1, -4.9, 4.1)),

        ((-0.9, -0.9), Area(-10.0, 10.0, 10.0, -10.0), Area(-10.9, 10.1, 10.1, -10.9)),
        ((-1000.9, 100.1), Area(-10.0, 10.0, 10.0, -10.0), Area(-10.9, 10.1, 10.1, -10.9)),
    ]
)
def test_anchor_offsets(anchor, area, expected):
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        make_vectors_with_id(42, {area}, path)

        source = VectorLayer.layer_from_file(
            path,
            "id_no = 42",
            PixelScale(1.0, -1.0),
            WGS_84_PROJECTION,
            anchor=anchor
        )

        final_area = source.area
        assert final_area == expected


@pytest.mark.parametrize(
    "klass",
    [
        VectorLayer,
        RasteredVectorLayer
    ]
)
def test_vector_layers_with_empty_features(klass) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 0.0, 0.0), 42),
            (Area(0.0, 0.0, 10, -10), 43)
        }
        make_vectors_with_empty_feature(areas, path)

        layer = klass.layer_from_file(path, None, PixelScale(1.0, -1.0), WGS_84_PROJECTION)

        assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 20)
        assert layer.datatype == DataType.Byte

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        assert total == (layer.window.xsize * layer.window.ysize) / 2
