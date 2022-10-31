import os
import tempfile

import h3
import pytest

from helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe.h3layer import H3CellLayer
from yirgacheffe.layers import Area, Layer, PixelScale, Window, VectorRangeLayer, DynamicVectorRangeLayer
from yirgacheffe.operators import LayerOperation, ShaderStyleOperation

WSG_84_PROJECTION = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'\
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'\
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],'\
    'AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

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

        layer = VectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        assert layer.area == area
        assert layer.geo_transform == (area.left, 1.0, 0.0, area.top, 0.0, -1.0)
        assert layer.window == Window(0, 0, 20, 10)
        assert layer.projection == WSG_84_PROJECTION

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

        layer = DynamicVectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
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
            _ = DynamicVectorRangeLayer(path, "id_no = 123", PixelScale(1.0, -1.0), WSG_84_PROJECTION)

def test_multi_area_vector() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.gpkg")
        areas = {
            Area(-10.0, 10.0, 0.0, 0.0),
            Area(0.0, 0.0, 10, -10)
        }
        make_vectors_with_id(42, areas, path)

        vector_layer = VectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
        dynamic_layer = DynamicVectorRangeLayer(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)

        for layer in (dynamic_layer, vector_layer):
            assert layer.area == Area(-10.0, 10.0, 10.0, -10.0)
            assert layer.geo_transform == (-10.0, 1.0, 0.0, 10.0, 0.0, -1.0)
            assert layer.window == Window(0, 0, 20, 20)

        window = vector_layer.window
        for yoffset in range(window.ysize):
            vector_raster = vector_layer.read_array(0, yoffset, window.xsize, 1)
            dynamic_raster = dynamic_layer.read_array(0, yoffset, window.xsize, 1)
            assert vector_raster.shape == (1, window.xsize)
            assert (vector_raster == dynamic_raster).all()

        del vector_layer
        del dynamic_layer


@pytest.mark.parametrize(
    "cell_id,is_valid,expected_zoom",
    [
        ("hello", False, 0),
        ("88972eac11fffff", True, 8),
    ]
)
def test_h3_layer(cell_id: str, is_valid: bool, expected_zoom: int) -> None:
    if is_valid:
        layer = H3CellLayer(cell_id, PixelScale(0.001, -0.001), WSG_84_PROJECTION)
        assert layer.zoom == expected_zoom
        assert layer.projection == WSG_84_PROJECTION

        # without getting too deep, we'd expect a mix of zeros and ones in the data
        window = layer.window
        one_count = 0
        for yoffset in range(window.ysize):
            data = layer.read_array(0, yoffset, window.xsize, 1)
            assert data.shape == (1, window.xsize)
            one_count += data.sum()
        assert one_count != 0

    else:
        with pytest.raises(ValueError):
            _ = H3CellLayer(cell_id, PixelScale(0.001, -0.001), WSG_84_PROJECTION)

@pytest.mark.parametrize(
    "lat,lng",
    [
        (0.0, 0.0),
        (0.0, 45.0),
        (45.0, 0.0),
        (45.0, 45.0),
        (85.0, 0.0),
        (85.0, 45.0),
        (1.0, 95.0),
    ]
)
def test_h3_layer_magnifications(lat: float, lng: float) -> None:
    for zoom in range(6, 10):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        h3_layer = H3CellLayer(cell_id, PixelScale(0.000898315284120,-0.000898315284120), WSG_84_PROJECTION)

        on_cell_count = LayerOperation(h3_layer).sum()
        total_count = ShaderStyleOperation(h3_layer, lambda _: 1).sum()

        assert total_count == (h3_layer.window.xsize * h3_layer.window.ysize)
        assert 0 < on_cell_count < total_count

@pytest.mark.parametrize(
    "lat,lng",
    [
        (0.0, 0.0),
        (0.0, 45.0),
        (45.0, 0.0),
        (45.0, 45.0),
        (85.0, 0.0),
        (85.0, 45.0),
        (1.0, 95.0),
    ]
)
def test_h3_layer_not_clipped(lat: float, lng: float) -> None:
    for zoom in range(6, 10):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        scale = PixelScale(0.000898315284120,-0.000898315284120)
        h3_layer = H3CellLayer(cell_id, scale, WSG_84_PROJECTION)

        on_cell_count = LayerOperation(h3_layer).sum()

        before_window = h3_layer.window
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
        expanded_area = Area(
            left=h3_layer.area.left - (2 * abs_xstep),
            top=h3_layer.area.top + (2 * abs_ystep),
            right=h3_layer.area.right + (2 * abs_xstep),
            bottom=h3_layer.area.bottom - (2 * abs_ystep)
        )
        h3_layer.set_window_for_union(expanded_area)
        assert h3_layer.window > before_window

        expanded_on_cell_count = LayerOperation(h3_layer).sum()
        assert expanded_on_cell_count == on_cell_count
