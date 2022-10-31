
import h3
import numpy as np
import pytest

from yirgacheffe.h3layer import H3CellLayer
from yirgacheffe.layers import PixelScale
from yirgacheffe.operators import LayerOperation

class NaiveH3CellLayer(H3CellLayer):
    """h3.latlng_to_cell is quite expensive when you call it thousands of times
    so the H3CellLayer has a bunch of tricks to try reduce the work done. This is a naive
    version that checks for every cell."""

    def read_array(self, xoffset, yoffset, xsize, ysize):
        res = np.zeros((ysize, xsize), dtype=float)
        start_x = self._intersection.left + (xoffset * self._transform[1])
        start_y = self._intersection.top + (yoffset * self._transform[5])
        for ypixel in range(ysize):
            lat = start_y + (ypixel * self._transform[5])
            for xpixel in range(xsize):
                lng = start_x + (xpixel * self._transform[1])
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
        optimised_layer = H3CellLayer(cell_id, PixelScale(0.000898315284120,-0.000898315284120), "NOTUSED")
        naive_layer = NaiveH3CellLayer(cell_id, PixelScale(0.000898315284120,-0.000898315284120), "NOTUSED")

        optimised_cell_count = LayerOperation(optimised_layer).sum()
        naive_cell_count = LayerOperation(naive_layer).sum()

        assert optimised_cell_count != 0
        assert optimised_cell_count == naive_cell_count
