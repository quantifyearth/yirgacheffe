import os
import tempfile

import pytest

from tests.unit.helpers import (
    make_vectors_with_multiple_ids,
    make_vectors_with_id,
    make_vectors_with_empty_feature,
)
from yirgacheffe import DataType
from yirgacheffe.layers import RasterLayer, VectorLayer
from yirgacheffe import Area, MapProjection, PixelScale
from yirgacheffe._datatypes import Window


def test_basic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            with VectorLayer.layer_from_file(
                path, "id_no = 123", PixelScale(1.0, -1.0), "WGS 84"
            ) as _layer:
                pass


def test_basic_dynamic_vector_layer() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with VectorLayer.layer_from_file(
            path, "id_no = 42", PixelScale(1.0, -1.0), "epsg:4326"
        ) as layer:
            assert layer.area == Area(
                -10.0, 10.0, 10.0, 0.0, MapProjection("epsg:4326", 1.0, -1.0)
            )
            assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
            assert layer.dimensions == (20, 10)
            assert layer._virtual_window == Window(0, 0, 20, 10)
            assert layer.map_projection.epsg == 4326

            # The astype here is to catch escaping MLX types...
            res = layer.read_array(0, 0, 20, 20).astype(int)
            assert res.sum() > 0


def test_basic_dynamic_vector_layer_no_filter_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        with pytest.raises(ValueError):
            _ = VectorLayer.layer_from_file(
                path, "id_no = 123", PixelScale(1.0, -1.0), "epsg:4326"
            )


def test_multi_area_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {Area(-10.0, 10.0, 0.0, 0.0), Area(0.0, 0.0, 10, -10)}
        make_vectors_with_id(42, areas, path)

        layer = VectorLayer.layer_from_file(
            path, "id_no = 42", PixelScale(1.0, -1.0), "epsg:4326"
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)

        xsize, ysize = layer.dimensions
        for yoffset in range(ysize):
            raster = layer.read_array(0, yoffset, xsize, 1)
            assert raster.shape == (1, xsize)


def test_empty_layer_from_vector():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(
            left=44.00253688814017,
            top=-12.440948032828079,
            right=50.483612168477286,
            bottom=-25.1535466075739,
        )
        make_vectors_with_id(42, {area}, path)

        source = VectorLayer.layer_from_file(
            path,
            "id_no = 42",
            PixelScale(xstep=0.00026949458523585647, ystep=-0.00026949458523585647),
            "epsg:4326",
        )

        empty = RasterLayer.empty_raster_layer_like(source)
        assert empty.map_projection == source.map_projection
        assert empty.dimensions == source.dimensions
        assert empty._virtual_window == source._virtual_window
        assert empty.area == source.area


def test_vector_layers_with_default_burn_value() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {(Area(-10.0, 10.0, 0.0, 0.0), 42), (Area(0.0, 0.0, 10, -10), 43)}
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path, None, PixelScale(1.0, -1.0), "epsg:4326"
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)
        assert layer.datatype == DataType.Byte

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (xsize * ysize) / 2


def test_vector_layers_with_fixed_burn_value() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {(Area(-10.0, 10.0, 0.0, 0.0), 42), (Area(0.0, 0.0, 10, -10), 43)}
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path, None, PixelScale(1.0, -1.0), "epsg:4326", burn_value=5
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == ((xsize * ysize) / 2) * 5


def test_vector_layers_with_default_burn_value_and_filter() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {(Area(-10.0, 10.0, 0.0, 0.0), 42), (Area(0.0, 0.0, 10, -10), 43)}
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path, "id_no=42", PixelScale(1.0, -1.0), "epsg:4326"
        )

        assert layer.area == Area(
            -10.0, 10.0, 0.0, 0.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (10, 10)
        assert layer._virtual_window == Window(0, 0, 10, 10)

        # Because we picked one later, all pixels should be burned
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (xsize * ysize)


def test_vector_layers_with_invalid_burn_value() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {(Area(-10.0, 10.0, 0.0, 0.0), 42), (Area(0.0, 0.0, 10, -10), 43)}
        make_vectors_with_multiple_ids(areas, path)

        with pytest.raises(ValueError):
            _ = VectorLayer.layer_from_file(
                path,
                None,
                PixelScale(1.0, -1.0),
                "epsg:4326",
                burn_value="this_is_wrong",
            )


def test_vector_layers_with_field_value() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {(Area(-10.0, 10.0, 0.0, 0.0), 42), (Area(0.0, 0.0, 10, -10), 43)}
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path, None, PixelScale(1.0, -1.0), "epsg:4326", burn_value="id_no"
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (((xsize * ysize) / 4) * 42) + (((xsize * ysize) / 4) * 43)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, DataType.Byte),
        (42, DataType.Byte),
        (-1, DataType.Int16),
        (1024, DataType.UInt16),
        (1024 * 1024, DataType.UInt32),
        (-1024 * 1024, DataType.Int32),
        (1.0, DataType.Float64),
    ],
)
def test_vector_layers_with_guessed_type_burn_value(value, expected) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 10.0, -10.0), value),
        }
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path, None, PixelScale(1.0, -1.0), "epsg:4326", burn_value=value
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)
        assert layer.datatype == expected

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (xsize * ysize) * value


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
    ],
)
def test_vector_layers_with_different_type_burn_value(value, datatype) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 10.0, -10.0), value),
        }
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path,
            None,
            PixelScale(1.0, -1.0),
            "epsg:4326",
            datatype=datatype,
            burn_value="id_no",
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (xsize * ysize) * value


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, DataType.Int64),
        (1.0, DataType.Float64),
    ],
)
def test_vector_layers_with_guess_field_type_burn_value(value, expected) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            (Area(-10.0, 10.0, 10.0, -10.0), value),
        }
        make_vectors_with_multiple_ids(areas, path)

        layer = VectorLayer.layer_from_file(
            path, None, PixelScale(1.0, -1.0), "epsg:4326", burn_value="id_no"
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)
        assert layer.datatype == expected

        # The default burn value is 1, so check that if we sum the area
        # we get half and half, but then multiplied by burn value
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (xsize * ysize) * value


@pytest.mark.parametrize(
    "size,expect_success",
    [
        ((5, 5), True),
        ((5, 1), True),
        ((1, 5), True),
        ((5, 0), False),
        ((0, 5), False),
    ],
)
def test_read_array_size(size, expect_success):
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        area = Area(-10.0, 10.0, 10.0, 0.0)
        make_vectors_with_id(42, {area}, path)

        source = VectorLayer.layer_from_file(
            path, "id_no = 42", PixelScale(1.0, -1.0), "epsg:4326"
        )

        if expect_success:
            data = source.read_array(0, 0, size[0], size[1])
            assert data.shape == (size[1], size[0])
        else:
            with pytest.raises(ValueError):
                _ = source.read_array(0, 0, size[0], size[1])


@pytest.mark.parametrize(
    "anchor,area,expected",
    [
        (
            (0.0, 0.0),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(-9.9, 9.9, 9.9, -9.9),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(-9.1, 9.1, 9.1, -9.1),
            Area(-10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(5.0, 10.0, 10.0, 5.0),
            Area(5.0, 10.0, 10.0, 5.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(5.0, -5.0, 10.0, -10.0),
            Area(5.0, -5.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(-10.0, -5.0, -5.0, -10.0),
            Area(-10.0, -5.0, -5.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(-10.0, 10.0, -5.0, 5.0),
            Area(-10.0, 10.0, -5.0, 5.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(5.1, 9.9, 9.9, 5.1),
            Area(5.0, 10.0, 10.0, 5.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(5.1, -5.1, 9.9, -9.9),
            Area(5.0, -5.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(-9.9, -5.1, -5.1, -9.9),
            Area(-10.0, -5.0, -5.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.0, 0.0),
            Area(-9.9, 9.9, -5.1, 5.1),
            Area(-10.0, 10.0, -5.0, 5.0, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.1, 0.1),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.9, 10.1, 10.1, -10.9, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.1, 0.1),
            Area(5.0, 10.0, 10.0, 5.0),
            Area(4.1, 10.1, 10.1, 4.1, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.1, 0.1),
            Area(5.0, -5.0, 10.0, -10.0),
            Area(4.1, -4.9, 10.1, -10.9, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.1, 0.1),
            Area(-10.0, -5.0, -5.0, -10.0),
            Area(-10.9, -4.9, -4.9, -10.9, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (0.1, 0.1),
            Area(-10.0, 10.0, -5.0, 5.0),
            Area(-10.9, 10.1, -4.9, 4.1, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (-0.9, -0.9),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.9, 10.1, 10.1, -10.9, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
        (
            (-1000.9, 100.1),
            Area(-10.0, 10.0, 10.0, -10.0),
            Area(-10.9, 10.1, 10.1, -10.9, MapProjection("epsg:4326", 1.0, -1.0)),
        ),
    ],
)
def test_anchor_offsets(anchor, area, expected):
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        make_vectors_with_id(42, {area}, path)

        source = VectorLayer.layer_from_file(
            path, "id_no = 42", PixelScale(1.0, -1.0), "epsg:4326", anchor=anchor
        )

        final_area = source.area
        assert final_area == expected


def test_vector_layers_with_empty_features() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {(Area(-10.0, 10.0, 0.0, 0.0), 42), (Area(0.0, 0.0, 10, -10), 43)}
        make_vectors_with_empty_feature(areas, path)

        layer = VectorLayer.layer_from_file(
            path, None, PixelScale(1.0, -1.0), "epsg:4326"
        )

        assert layer.area == Area(
            -10.0, 10.0, 10.0, -10.0, MapProjection("epsg:4326", 1.0, -1.0)
        )
        assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
        assert layer.dimensions == (20, 20)
        assert layer._virtual_window == Window(0, 0, 20, 20)
        assert layer.datatype == DataType.Byte

        # The default burn value is 1, so check that if we sum the area
        # we get half and half
        total = layer.sum()
        xsize, ysize = layer.dimensions
        assert total == (xsize * ysize) / 2
