import h3
import numpy as np
import pytest
from osgeo import gdal

from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import RasterLayer, H3CellLayer
from yirgacheffe.window import Area, MapProjection
from yirgacheffe._backends import backend

# work around of pylint
demote_array = backend.demote_array


@pytest.mark.parametrize(
    "cell_id,is_valid,expected_zoom",
    [
        ("hello", False, 0),
        ("88972eac11fffff", True, 8),
    ]
)
def test_h3_layer(cell_id: str, is_valid: bool, expected_zoom: int) -> None:
    if is_valid:
        with H3CellLayer(cell_id, MapProjection(WGS_84_PROJECTION, 0.001, -0.001)) as layer:
            assert layer.zoom == expected_zoom
            assert layer.projection == WGS_84_PROJECTION

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
            with H3CellLayer(cell_id, MapProjection(WGS_84_PROJECTION, 0.001, -0.001)) as _layer:
                pass

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
        h3_layer = H3CellLayer(
            cell_id,
            MapProjection(WGS_84_PROJECTION, 0.000898315284120,-0.000898315284120)
        )
        on_cell_count = h3_layer.sum()
        total_count = h3_layer.window.xsize * h3_layer.window.ysize
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
        projection = MapProjection(
            WGS_84_PROJECTION,
            0.000898315284120,
            -0.000898315284120
        )
        h3_layer = H3CellLayer(cell_id, projection)

        on_cell_count = h3_layer.sum()
        assert on_cell_count > 0.0

        before_window = h3_layer.window
        abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)
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
        projection = MapProjection(
            WGS_84_PROJECTION,
            0.000898315284120,
            -0.000898315284120
        )
        h3_layer = H3CellLayer(cell_id, projection)

        on_cell_count = h3_layer.sum()
        assert on_cell_count > 0.0

        before_window = h3_layer.window
        abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)
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
        assert np.sum(demote_array(h3_layer.read_array(0, 0, h3_layer.window.xsize, 5))) > 0.0
        assert np.sum(demote_array(h3_layer.read_array(0, h3_layer.window.ysize - 5, h3_layer.window.xsize, 5))) > 0.0
        assert np.sum(demote_array(h3_layer.read_array(0, 0, 5, h3_layer.window.ysize))) > 0.0
        assert np.sum(demote_array(h3_layer.read_array(h3_layer.window.xsize - 5, 0, 5, h3_layer.window.ysize))) > 0.0

@pytest.mark.parametrize(
    "lat,lng",
    [
        (50.0, 179.9),
        (50.0, -179.9),
    ]
)
def test_h3_layer_wrapped_on_projection(lat: float, lng: float) -> None:
    cell_id = h3.latlng_to_cell(lat, lng, 3)
    projection = MapProjection(WGS_84_PROJECTION, 0.01, -0.01)
    h3_layer = H3CellLayer(cell_id, projection)

    # Just sanity check this test has caught a cell that wraps the entire planet and is testing
    # what we think it is testing:
    assert h3_layer.window.xsize == (360 / 0.01)

    area = h3_layer.sum()
    assert area > 0.0 # sanity check

    # Go around the cell neighbours, of which some will not wrap the planet, and
    # check they are all of a similarish size - we had a bug early on where we'd
    # mistakenly invert the area for the band, counting all the cells across the planet
    for cell_id in h3.grid_ring(cell_id, 1):
        neighbour = H3CellLayer(cell_id, projection)
        neighbour_area = neighbour.sum()
        # We're happy if they're within 10% for now
        assert abs((neighbour_area - area) / area) < 0.1


    before_window = h3_layer.window
    _, abs_ystep = abs(projection.xstep), abs(projection.ystep)
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
    assert np.sum(demote_array(h3_layer.read_array(0, 0, h3_layer.window.xsize, 2))) == 0.0
    assert np.sum(demote_array(h3_layer.read_array(0, h3_layer.window.ysize - 2, h3_layer.window.xsize, 2))) == 0.0

def test_h3_layer_overlapped():
    # This is based on a regression, where somehow I made tiles not tesselate properly
    left, top = (121.26706, 19.45338)
    right, bottom = (121.62494, 19.18478)
    projection = MapProjection(
        WGS_84_PROJECTION,
        0.000898315284120,
        -0.000898315284120
    )

    cells = h3.geo_to_cells(h3.LatLngPoly([
        (top, left),
        (top, right),
        (bottom, right),
        (bottom, left),
    ],), 7)

    tiles = [
        H3CellLayer(cell_id, projection)
    for cell_id in cells]

    union = RasterLayer.find_union(tiles)
    union = union.grow(0.02)

    scratch = RasterLayer.empty_raster_layer(union, projection.scale, gdal.GDT_Float64)

    # In scratch we should only have 0 or 1 values, but if there are any overlaps we should get 2s...

    for tile in tiles:
        scratch.reset_window()
        layers = [scratch, tile]
        intersection = RasterLayer.find_intersection(layers)
        for layer in layers:
            layer.set_window_for_intersection(intersection)
        result = scratch + tile
        result.save(scratch)

    scratch.reset_window()
    calc = scratch > 1.5
    assert calc.sum() == 0
