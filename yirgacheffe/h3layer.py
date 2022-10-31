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

        if ((xoffset + xsize) > self.window.xsize) or \
            ((yoffset + ysize) > self.window.ysize) or \
            (xoffset < 0) or \
            (yoffset < 0):
            raise ValueError("Request area goes out of bounds")

        res = np.zeros((ysize, xsize), dtype=float)

        # this all feels very WGS84 specific? I could also throw the h3
        # polygon into OGR and just do the same logic as DynamicVectorTileArray, but if
        # we are in the usual mode, this code should be faster.

        # trim the interesting bit of x
        if self.window.xoff < 0:
            xoffset -= self.window.xoff
            xsize += self.window.xoff
        if (xsize - xoffset) > self._raster_xsize:
            xsize = self._raster_xsize - xoffset

        for ypixel in range(yoffset, yoffset + ysize):

            # The latlng_to_cell is quite expensive, so we could in from either side
            # and then fill. A binary search probably would be nicer...
            left_most = xoffset + xsize + 1
            right_most = -1

            lat = self._intersection.top + (ypixel * self._transform[5])

            for xpixel in range(xoffset, xoffset + xsize):
                lng = self._intersection.left + (xpixel * self._transform[1])
                this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                if this_cell == self.cell_id:
                    left_most = xpixel
                    break

            for xpixel in range(xoffset + xsize - 1, left_most - 1, -1):
                lng = self._intersection.left + (xpixel * self._transform[1])
                this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                if this_cell == self.cell_id:
                    right_most = xpixel
                    break

            for xpixel in range(left_most, right_most + 1, 1):
                res[ypixel - yoffset][xpixel - xoffset] = 1.0

        return res
