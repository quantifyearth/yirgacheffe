from math import ceil, floor

import h3
import numpy as np

from .layers import Layer, PixelScale
from .window import Area, Window

class H3CellLayer(Layer):

    def __init__(self, cell_id: str, scale: PixelScale, projection: str):
        if not h3.is_valid_cell(cell_id):
            raise ValueError(f"{cell_id} is not a valid H3 cell identifier")
        self.cell_id = cell_id
        self.zoom = h3.get_resolution(cell_id)

        self.cell_boundary = h3.cell_to_boundary(cell_id)

        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
        self.area = Area(
            left=(floor(min(x[1] for x in self.cell_boundary) / abs_xstep) * abs_xstep),
            top=(ceil(max(x[0] for x in self.cell_boundary) / abs_ystep) * abs_ystep),
            right=(ceil(max(x[1] for x in self.cell_boundary) / abs_xstep) * abs_xstep),
            bottom=(floor(min(x[0] for x in self.cell_boundary) / abs_ystep) * abs_ystep),
        )
        self._transform = [self.area.left, scale.xstep, 0.0, self.area.top, 0.0, scale.ystep]
        self._projection = projection
        self._dataset = None
        self._intersection = self.area
        self.window = Window(
            xoff=0,
            yoff=0,
            xsize=ceil((self.area.right - self.area.left) / scale.xstep),
            ysize=ceil((self.area.bottom - self.area.top) / scale.ystep),
        )
        self._raster_xsize = self.window.xsize
        self._raster_ysize = self.window.ysize

    @property
    def projection(self) -> str:
        return self._projection

    def read_array(self, xoffset, yoffset, xsize, ysize):

        # there's some optimisation here were we could check if the requested area
        # is in the polygon
        if (yoffset + self.window.yoff < 0) or (((yoffset + ysize) + self.window.yoff) > self._raster_ysize):
            return 0.0

        res = np.zeros((ysize, xsize), dtype=float)

        # this all feels very WGS84 specific? I could also throw the h3
        # polygon into OGR and just do the same logic as DynamicVectorTileArray, but if
        # we are in the usual mode, this code should be faster.

        start_x = self._intersection.left + (xoffset * self._transform[1])
        start_y = self._intersection.top + (yoffset * self._transform[5])

        for ypixel in range(ysize):

            # The latlng_to_cell is quite expensive, so we could in from either side
            # and then fill. A binary search probably would be nicer...
            left_most = xsize + 1
            right_most = -1

            lat = start_y + (ypixel * self._transform[5])
            for xpixel in range(xsize):
                lng = start_x + (xpixel * self._transform[1])
                this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                if this_cell == self.cell_id:
                    left_most = xpixel
                    break

            for xpixel in range(xsize, 0, -1):
                lng = start_x + (xpixel * self._transform[1])
                this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                if this_cell == self.cell_id:
                    right_most = xpixel
                    break

            for xpixel in range(left_most, right_most + 1, 1):
                res[ypixel][xpixel] = 1.0

        return res
