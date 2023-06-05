import os
import tempfile

import h3
import numpy as np
import pytest
from osgeo import gdal

from helpers import gdal_dataset_of_region, make_vectors_with_id
from yirgacheffe import WSG_84_PROJECTION
from yirgacheffe.h3layer import H3CellLayer
from yirgacheffe.layers import Area, RasterLayer, PixelScale, Window, VectorRangeLayer, DynamicVectorRangeLayer, RasteredVectorLayer, VectorLayer
from yirgacheffe.rounding import round_up_pixels
from yirgacheffe.operators import ShaderStyleOperation


# There is a lot of "del" in this file, due to a combination of gdal having no way
# to explicitly close a file beyond forcing the gdal object's deletion, and Windows
# getting upset that it tries to clear up the TemporaryDirectory and there's an open
# file handle within that directory.

def test_make_basic_layer() -> None:
    area = Area(-10, 10, 10, -10)
    layer = RasterLayer(gdal_dataset_of_region(area, 0.02))
    assert layer.area == area
    assert layer.pixel_scale == (0.02, -0.02)
    assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
    assert layer.window == Window(0, 0, 1000, 1000)

def test_make_basic_layer_old_name() -> None:
    from yirgacheffe.layers import Layer

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
        _ = RasterLayer(None)

def test_layer_from_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = RasterLayer.layer_from_file("this_file_does_not_exist.tif")

def test_open_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        del dataset
        assert os.path.exists(path)
        layer = RasterLayer.layer_from_file(path)
        assert layer.area == area
        assert layer.pixel_scale == (0.02, -0.02)
        assert layer.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
        assert layer.window == Window(0, 0, 1000, 1000)
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

        rastered_layer = RasteredVectorLayer.layer_from_file(path, "id_no = 42", PixelScale(1.0, -1.0), WSG_84_PROJECTION)
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

        on_cell_count = h3_layer.sum()
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

        on_cell_count = h3_layer.sum()
        assert on_cell_count > 0.0

        before_window = h3_layer.window
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
        expanded_area = Area(
            left=h3_layer.area.left - (12 * abs_xstep),
            top=h3_layer.area.top + (12 * abs_ystep),
            right=h3_layer.area.right + (12 * abs_xstep),
            bottom=h3_layer.area.bottom - (12 * abs_ystep)
        )
        h3_layer.set_window_for_union(expanded_area)
        assert h3_layer.window > before_window

        expanded_on_cell_count = h3_layer.sum()
        assert expanded_on_cell_count == on_cell_count

        # whilst we're here, check that we do have an empty border (i.e., does window for union do the right thing)
        assert np.sum(h3_layer.read_array(0, 0, h3_layer.window.xsize, 2)) == 0.0
        assert np.sum(h3_layer.read_array(0, h3_layer.window.ysize - 2, h3_layer.window.xsize, 2)) == 0.0
        assert np.sum(h3_layer.read_array(0, 0, 2, h3_layer.window.ysize)) == 0.0
        assert np.sum(h3_layer.read_array(h3_layer.window.xsize - 2, 0, 2, h3_layer.window.ysize)) == 0.0

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
def test_h3_layer_clipped(lat: float, lng: float) -> None:
    for zoom in range(6, 8):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        scale = PixelScale(0.000898315284120,-0.000898315284120)
        h3_layer = H3CellLayer(cell_id, scale, WSG_84_PROJECTION)

        on_cell_count = h3_layer.sum()
        assert on_cell_count > 0.0

        before_window = h3_layer.window
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
        expanded_area = Area(
            left=h3_layer.area.left + (2 * abs_xstep),
            top=h3_layer.area.top - (2 * abs_ystep),
            right=h3_layer.area.right - (2 * abs_xstep),
            bottom=h3_layer.area.bottom + (2 * abs_ystep)
        )
        h3_layer.set_window_for_intersection(expanded_area)
        assert h3_layer.window < before_window

        shrunk_on_cell_count = h3_layer.sum()
        assert shrunk_on_cell_count < on_cell_count

        # whilst we're here, check that we do have an filled border (i.e., does window for
        # intersection do the right thing)
        assert np.sum(h3_layer.read_array(0, 0, h3_layer.window.xsize, 5)) > 0.0
        assert np.sum(h3_layer.read_array(0, h3_layer.window.ysize - 5, h3_layer.window.xsize, 5)) > 0.0
        assert np.sum(h3_layer.read_array(0, 0, 5, h3_layer.window.ysize)) > 0.0
        assert np.sum(h3_layer.read_array(h3_layer.window.xsize - 5, 0, 5, h3_layer.window.ysize)) > 0.0

@pytest.mark.parametrize(
    "lat,lng",
    [
        (50.0, 179.9),
        (50.0, -179.9),
    ]
)
def test_h3_layer_wrapped_on_projection(lat: float, lng: float) -> None:
    cell_id = h3.latlng_to_cell(lat, lng, 3)
    scale = PixelScale(0.01, -0.01)
    h3_layer = H3CellLayer(cell_id, scale, WSG_84_PROJECTION)

    # Just sanity check this test has caught a cell that wraps the entire planet and is testing
    # what we think it is testing:
    assert h3_layer.window.xsize == (360 / 0.01)

    area = h3_layer.sum()
    assert area > 0.0 # sanity check

    # Go around the cell neighbours, of which some will not wrap the planet, and
    # check they are all of a similarish size - we had a bug early on where we'd
    # mistakenly invert the area for the band, counting all the cells across the planet
    for cell_id in h3.grid_ring(cell_id, 1):
        neighbour = H3CellLayer(cell_id, scale, WSG_84_PROJECTION)
        neighbour_area = neighbour.sum()
        # We're happy if they're within 10% for now
        assert abs((neighbour_area - area) / area) < 0.1


    before_window = h3_layer.window
    _, abs_ystep = abs(scale.xstep), abs(scale.ystep)
    expanded_area = Area(
        left=h3_layer.area.left,
        top=h3_layer.area.top + (22 * abs_ystep),
        right=h3_layer.area.right,
        bottom=h3_layer.area.bottom - (22 * abs_ystep)
    )
    h3_layer.set_window_for_union(expanded_area)
    assert h3_layer.window.xsize == before_window.xsize
    assert h3_layer.window.xoff == 0
    assert before_window.xoff == 0
    assert h3_layer.window.yoff < 0
    assert before_window.yoff == 0
    assert h3_layer.window.ysize > before_window.ysize

    expanded_area = h3_layer.sum()
    assert expanded_area == area

    # whilst we're here, check that we do have an empty border (i.e., does window for union do the right thing)
    assert np.sum(h3_layer.read_array(0, 0, h3_layer.window.xsize, 2)) == 0.0
    assert np.sum(h3_layer.read_array(0, h3_layer.window.ysize - 2, h3_layer.window.xsize, 2)) == 0.0

@pytest.mark.parametrize("initial_area",
    [
        Area(left=10.0, top=10.0, right=10.1, bottom=9.9),
        Area(left=-10.0, top=10.0, right=-9.9, bottom=9.9),
        Area(left=10.0, top=-10.0, right=10.1, bottom=-10.1),
        Area(left=-10.0, top=-10.0, right=-9.9, bottom=-10.1),
        Area(left=-0.1, top=0.1, right=0.1, bottom=-0.1),
    ]
)
def test_empty_layers_are_pixel_aligned(initial_area):
    scale = PixelScale(0.000898315284120,-0.000898315284120)

    expanded_area = initial_area.grow(0.1)

    initial_layer = RasterLayer.empty_raster_layer(initial_area, scale, gdal.GDT_Float64)
    print((initial_layer.area.right - initial_layer.area.left) / scale.xstep)

    pixel_width = (initial_layer.area.right - initial_layer.area.left) / scale.xstep
    assert round_up_pixels(pixel_width, scale.xstep) == initial_layer.window.xsize

    expanded_layer = RasterLayer.empty_raster_layer(expanded_area, scale, gdal.GDT_Float64)
    pixel_width = (expanded_layer.area.right - expanded_layer.area.left) / scale.xstep
    assert round_up_pixels(pixel_width, scale.xstep) == expanded_layer.window.xsize


def test_h3_layer_overlapped():
    # This is based on a regression, where somehow I made tiles not tesselate properly
    left, top = (121.26706, 19.45338)
    right, bottom = (121.62494, 19.18478)
    scale = PixelScale(0.000898315284120,-0.000898315284120)

    cells = h3.polygon_to_cells(h3.Polygon([
        (top, left),
        (top, right),
        (bottom, right),
        (bottom, left),
    ],), 7)

    tiles = [
        H3CellLayer(cell_id, scale, WSG_84_PROJECTION)
    for cell_id in cells]

    union = RasterLayer.find_union(tiles)
    union = union.grow(0.02)

    scratch = RasterLayer.empty_raster_layer(union, scale, gdal.GDT_Float64)

    # In scratch we should only have 0 or 1 values, but if there are any overlaps we should get 2s...

    for tile in tiles:
        scratch.reset_window()
        layers = [scratch, tile]
        intersection = RasterLayer.find_intersection(layers)
        for layer in layers:
            layer.set_window_for_intersection(intersection)
        result = scratch + tile
        result.save(scratch)

    def overlap_check(chunk):
        # if we have rounding errors, then cells will overlap
        # and we'll end up with values higher than 1.0 in the cell
        # which would leave to double accounting
        if np.sum(chunk > 1.5):
            raise Exception
        return chunk

    scratch.reset_window()
    calc = scratch.numpy_apply(overlap_check)
    calc.save(scratch)

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

def test_empty_layer_from_raster():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    empty = RasterLayer.empty_raster_layer_like(source)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.window == source.window
    assert empty.datatype == source.datatype

def test_empty_layer_from_raster_new_datatype():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.datatype == gdal.GDT_Byte
    empty = RasterLayer.empty_raster_layer_like(source, datatype=gdal.GDT_Float64)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.window == source.window
    assert empty.datatype == gdal.GDT_Float64

def test_empty_layer_from_raster_with_window():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    original_window = source.window

    source.set_window_for_intersection(Area(-1, 1, 1, -1))
    assert source.window < original_window

    empty = RasterLayer.empty_raster_layer_like(source)
    assert empty.pixel_scale == source.pixel_scale
    assert empty.projection == source.projection
    assert empty.window.xoff == 0
    assert empty.window.yoff == 0
    assert empty.window.xsize == source.window.xsize
    assert empty.window.ysize == source.window.ysize
