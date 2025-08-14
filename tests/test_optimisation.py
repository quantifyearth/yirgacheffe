
import h3
import numpy as np
import pytest

from yirgacheffe import WGS_84_PROJECTION
from yirgacheffe.layers import RasterLayer, H3CellLayer
from yirgacheffe.window import Area, MapProjection
import yirgacheffe.operators as yo

class NaiveH3CellLayer(H3CellLayer):
    """h3.latlng_to_cell is quite expensive when you call it thousands of times
    so the H3CellLayer has a bunch of tricks to try reduce the work done. This is a naive
    version that checks for every cell."""

    def read_array(self, xoffset, yoffset, xsize, ysize): # pylint: disable=W0237
        assert self._projection is not None
        res = np.zeros((ysize, xsize), dtype=float)
        start_x = self.area.left + (xoffset * self._projection.xstep)
        start_y = self.area.top + (yoffset * self._projection.ystep)
        for ypixel in range(ysize):
            lat = start_y + (ypixel * self._projection.ystep)
            for xpixel in range(xsize):
                lng = start_x + (xpixel * self._projection.xstep)
                this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                if this_cell == self.cell_id:
                    res[ypixel][xpixel] = 1.0
        return res

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
def test_h3_vs_naive(lat: float, lng: float) -> None:
    for zoom in range(5, 9):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        projection = MapProjection(WGS_84_PROJECTION, 0.000898315284120,-0.000898315284120)
        optimised_layer = H3CellLayer(cell_id, projection)
        naive_layer = NaiveH3CellLayer(cell_id, projection)

        optimised_cell_count = optimised_layer.sum()
        naive_cell_count = naive_layer.sum()

        assert optimised_cell_count != 0
        assert optimised_cell_count == naive_cell_count

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
def test_h3_vs_naive_for_union(lat: float, lng: float) -> None:
    for zoom in range(7, 9):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        projection = MapProjection(WGS_84_PROJECTION, 0.000898315284120,-0.000898315284120)
        optimised_layer = H3CellLayer(cell_id, projection)
        naive_layer = NaiveH3CellLayer(cell_id, projection)

        before_cell_count = optimised_layer.sum()

        superset_area = Area(
            left=optimised_layer.area.left - (5 * projection.xstep),
            right=optimised_layer.area.right + (5 * projection.xstep),
            top=optimised_layer.area.top - (5 * projection.ystep),
            bottom=optimised_layer.area.bottom + (5 * projection.ystep),
        )
        optimised_layer.set_window_for_union(superset_area)
        naive_layer.set_window_for_union(superset_area)

        optimised_cell_count = optimised_layer.sum()
        naive_cell_count = naive_layer.sum()

        assert optimised_cell_count == before_cell_count
        assert optimised_cell_count == naive_cell_count

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
def test_h3_vs_naive_for_intersection(lat: float, lng: float) -> None:
    for zoom in range(7, 9):
        cell_id = h3.latlng_to_cell(lat, lng, zoom)
        projection = MapProjection(WGS_84_PROJECTION, 0.000898315284120,-0.000898315284120)
        optimised_layer = H3CellLayer(cell_id, projection)
        naive_layer = NaiveH3CellLayer(cell_id, projection)

        before_cell_count = optimised_layer.sum()

        subset_area = Area(
            left=optimised_layer.area.left + (2 * projection.xstep),
            right=optimised_layer.area.right - (2 * projection.xstep),
            top=optimised_layer.area.top + (2 * projection.ystep),
            bottom=optimised_layer.area.bottom - (2 * projection.ystep),
        )
        optimised_layer.set_window_for_intersection(subset_area)
        naive_layer.set_window_for_intersection(subset_area)

        optimised_cell_count = optimised_layer.sum()
        naive_cell_count = naive_layer.sum()

        assert optimised_cell_count < before_cell_count
        assert optimised_cell_count == naive_cell_count

# This test exits because certain hexagons have convex edges due to projection
# distortion. This means that my original naive attempt to speed up rendering by
# just finding the left and right most pixels on a line and filling between them
# leads to over-filling some tiles, which then overlap their neighbours.
@pytest.mark.parametrize(
    "cell_id",
    [
        "87a968290ffffff",
        "87a9680e0ffffff",
        "87a96811cffffff",
        "87a968293ffffff",
        "87a968768ffffff",
        "87a968893ffffff",
        "87a968000ffffff",
        "87a96b934ffffff",
        "87a968130ffffff",
    ]
)
def test_cells_dont_overlap(cell_id):

    cluster = h3.grid_disk(cell_id, 1)
    projection = MapProjection(WGS_84_PROJECTION, 0.000898315284120,-0.000898315284120)
    layers = [H3CellLayer(x, projection) for x in cluster]

    union = RasterLayer.find_union(layers)
    for layer in layers:
        layer.set_window_for_union(union)

    calc = layers[0]
    for layer in layers[1:]:
        calc += layer
    calc = yo.where(calc > 1, 1, 0)
    assert calc.sum() == 0

    # If we didn't get an exception, then there's no over drawing
