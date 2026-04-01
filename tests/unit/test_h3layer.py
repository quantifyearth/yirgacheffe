import h3
import numpy as np
import pytest
from osgeo import gdal

import yirgacheffe as yg
from yirgacheffe._layers import RasterLayer, H3CellLayer
from yirgacheffe import Area, MapProjection
from yirgacheffe._backends import backend

# work around of pylint
demote_array = backend.demote_array


@pytest.mark.parametrize(
    "cell_id,is_valid,expected_zoom",
    [
        ("hello", False, 0),
        ("88972eac11fffff", True, 8),
    ],
)
def test_h3_layer(cell_id: str, is_valid: bool, expected_zoom: int) -> None:
    if is_valid:
        with H3CellLayer(
            cell_id, MapProjection("epsg:4326", 0.001, -0.001)
        ) as layer:
            assert layer.zoom == expected_zoom
            assert layer.map_projection.epsg == 4326

            # without getting too deep, we'd expect a mix of zeros and ones in the data
            window = layer._virtual_window
            one_count = 0
            for yoffset in range(window.ysize):
                data = layer.read_array(0, yoffset, window.xsize, 1)
                assert data.shape == (1, window.xsize)
                one_count += data.sum()
            assert one_count != 0
    else:
        with pytest.raises(ValueError):
            with H3CellLayer(
                cell_id, MapProjection("epsg:4326", 0.001, -0.001)
            ) as _layer:
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
    ],
)
def test_h3_layer_magnifications(lat: float, lng: float) -> None:
    for zoom in range(6, 10):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        h3_layer = H3CellLayer(
            cell_id,
            MapProjection("epsg:4326", 0.000898315284120, -0.000898315284120),
        )
        on_cell_count = h3_layer.sum()
        total_count = h3_layer._virtual_window.xsize * h3_layer._virtual_window.ysize
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
    ],
)
def test_h3_layer_not_clipped(lat: float, lng: float) -> None:
    for zoom in range(6, 10):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        projection = MapProjection(
            "epsg:4326", 0.000898315284120, -0.000898315284120
        )
        h3_layer = H3CellLayer(cell_id, projection)

        on_cell_count = h3_layer.sum()
        assert on_cell_count > 0.0

        abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)
        expanded_area = Area(
            left=h3_layer.area.left - (12 * abs_xstep),
            top=h3_layer.area.top + (12 * abs_ystep),
            right=h3_layer.area.right + (12 * abs_xstep),
            bottom=h3_layer.area.bottom - (12 * abs_ystep),
            projection=projection,
        )
        expanded_layer = h3_layer.as_area(expanded_area)
        assert expanded_layer.dimensions > h3_layer.dimensions

        expanded_on_cell_count = expanded_layer.sum()
        assert expanded_on_cell_count == on_cell_count

        # whilst we're here, check that we do have an empty border (i.e., does window for union do the right thing)
        xsize, ysize = expanded_layer.dimensions
        assert np.sum(expanded_layer.read_array(0, 0, xsize, 2)) == 0.0
        assert (
            np.sum(
                expanded_layer.read_array(
                    0, ysize - 2, xsize, 2
                )
            )
            == 0.0
        )
        assert np.sum(expanded_layer.read_array(0, 0, 2, ysize)) == 0.0
        assert (
            np.sum(
                expanded_layer.read_array(
                    xsize - 2, 0, 2, ysize
                )
            )
            == 0.0
        )


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
    ],
)
def test_h3_layer_clipped(lat: float, lng: float) -> None:
    for zoom in range(6, 8):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        projection = MapProjection(
            "epsg:4326", 0.000898315284120, -0.000898315284120
        )
        h3_layer = H3CellLayer(cell_id, projection)

        on_cell_count = h3_layer.sum()
        assert on_cell_count > 0.0

        abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)
        expanded_area = Area(
            left=h3_layer.area.left + (2 * abs_xstep),
            top=h3_layer.area.top - (2 * abs_ystep),
            right=h3_layer.area.right - (2 * abs_xstep),
            bottom=h3_layer.area.bottom + (2 * abs_ystep),
            projection=projection,
        )

        shrunk_layer = h3_layer.as_area(expanded_area)
        assert shrunk_layer.dimensions < h3_layer.dimensions

        shrunk_on_cell_count = shrunk_layer.sum()
        assert shrunk_on_cell_count < on_cell_count

        # whilst we're here, check that we do have an filled border (i.e., does window for
        # intersection do the right thing)
        xsize, ysize = shrunk_layer.dimensions
        assert (
            np.sum(demote_array(shrunk_layer.read_array(0, 0, xsize, 5)))
            > 0.0
        )
        assert (
            np.sum(
                demote_array(
                    shrunk_layer.read_array(
                        0, ysize - 5, xsize, 5
                    )
                )
            )
            > 0.0
        )
        assert (
            np.sum(demote_array(shrunk_layer.read_array(0, 0, 5, ysize)))
            > 0.0
        )
        assert (
            np.sum(
                demote_array(
                    shrunk_layer.read_array(
                        xsize - 5, 0, 5, ysize
                    )
                )
            )
            > 0.0
        )


@pytest.mark.parametrize(
    "lat,lng",
    [
        (50.0, 179.9),
        (50.0, -179.9),
    ],
)
def test_h3_layer_wrapped_on_projection_and_expand(lat: float, lng: float) -> None:
    cell_id = h3.latlng_to_cell(lat, lng, 3)
    projection = MapProjection("epsg:4326", 0.01, -0.01)
    h3_layer = H3CellLayer(cell_id, projection)

    # Just sanity check this test has caught a cell that wraps the entire planet and is testing
    # what we think it is testing:
    assert h3_layer._virtual_window.xsize == (360 / 0.01)

    area = h3_layer.sum()
    assert area > 0.0  # sanity check

    # Go around the cell neighbours, of which some will not wrap the planet, and
    # check they are all of a similarish size - we had a bug early on where we'd
    # mistakenly invert the area for the band, counting all the cells across the planet
    for cell_id in h3.grid_ring(cell_id, 1):
        neighbour = H3CellLayer(cell_id, projection)
        neighbour_area = neighbour.sum()
        # We're happy if they're within 10% for now
        assert abs((neighbour_area - area) / area) < 0.1

    abs_ystep = abs(projection.ystep)
    expanded_area = Area(
        left=h3_layer.area.left,
        top=h3_layer.area.top + (22 * abs_ystep),
        right=h3_layer.area.right,
        bottom=h3_layer.area.bottom - (22 * abs_ystep),
        projection=projection,
    )
    expanded_layer = h3_layer.as_area(expanded_area)
    assert expanded_layer.dimensions[0] == h3_layer.dimensions[0]
    assert expanded_layer.dimensions[1] == h3_layer.dimensions[1] + 44

    expanded_area = expanded_layer.sum()
    assert expanded_area == area

    # whilst we're here, check that we do have an empty border (i.e., does window for union do the right thing)
    xsize, ysize = expanded_layer.dimensions
    assert np.sum(demote_array(expanded_layer.read_array(0, 0, xsize, 2))) == 0.0
    assert np.sum(demote_array(expanded_layer.read_array(0, ysize - 2, xsize, 2))) == 0.0


@pytest.mark.parametrize(
    "lat,lng",
    [
        (50.0, 179.9),
        (50.0, -179.9),
    ],
)
def test_h3_layer_wrapped_on_projection_and_shrink(lat: float, lng: float) -> None:
    cell_id = h3.latlng_to_cell(lat, lng, 3)
    projection = MapProjection("epsg:4326", 0.01, -0.01)
    h3_layer = H3CellLayer(cell_id, projection)

    # Just sanity check this test has caught a cell that wraps the entire planet and is testing
    # what we think it is testing:
    assert h3_layer._virtual_window.xsize == (360 / 0.01)

    area = h3_layer.sum()
    assert area > 0.0  # sanity check

    # Go around the cell neighbours, of which some will not wrap the planet, and
    # check they are all of a similarish size - we had a bug early on where we'd
    # mistakenly invert the area for the band, counting all the cells across the planet
    for cell_id in h3.grid_ring(cell_id, 1):
        neighbour = H3CellLayer(cell_id, projection)
        neighbour_area = neighbour.sum()
        # We're happy if they're within 10% for now
        assert abs((neighbour_area - area) / area) < 0.1

    abs_ystep = abs(projection.ystep)
    contacted_area = Area(
        left=h3_layer.area.left,
        top=h3_layer.area.top - (10 * abs_ystep),
        right=h3_layer.area.right,
        bottom=h3_layer.area.bottom + (10 * abs_ystep),
        projection=projection,
    )
    expanded_layer = h3_layer.as_area(contacted_area)
    assert expanded_layer.dimensions[0] == h3_layer.dimensions[0]
    assert expanded_layer.dimensions[1] == h3_layer.dimensions[1] - 20

    expanded_area = expanded_layer.sum()
    assert expanded_area < area

    # whilst we're here, check that we do don't have an empty border
    xsize, ysize = expanded_layer.dimensions
    assert np.sum(demote_array(expanded_layer.read_array(0, 0, xsize, 2))) != 0.0
    assert np.sum(demote_array(expanded_layer.read_array(0, ysize - 2, xsize, 2))) != 0.0


def test_h3_layer_overlapped():
    # This is based on a regression, where somehow I made tiles not tesselate properly
    left, top = (121.26706, 19.45338)
    right, bottom = (121.62494, 19.18478)
    projection = MapProjection("epsg:4326", 0.000898315284120, -0.000898315284120)

    cells = h3.geo_to_cells(
        h3.LatLngPoly(
            [
                (top, left),
                (top, right),
                (bottom, right),
                (bottom, left),
            ],
        ),
        7,
    )

    tiles = [H3CellLayer(cell_id, projection) for cell_id in cells]

    # In tile_sum we should only have 0 or 1 values, but if there are any overlaps we should get 2s...
    tile_sum = yg.sum(tiles)
    calc = tile_sum > 1.5
    assert calc.sum() == 0
