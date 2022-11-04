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

        # Statistically, most hex tiles will be within a projection, but some will wrap around
        # the edge, so check if we're hit one of those cases.
        self.centre = h3.cell_to_latlng(cell_id)

        # Due to time constraints, and that most geospatial data we are working with stops before the poles
        # I'm going to explicitly not handle the poles right now, where you get different distortions from
        # at the 180 line:
        if not self.area.bottom < self.centre[0] < self.area.top:
            raise NotImplementedError("Distortion at poles not currently handled")

        # For wrap around due to hitting the longitunal wrap, we currently just do the naive thing and
        # project a band for the full width of the projection. I have a plan to fix this, as I'd like layers
        # to have sublayers, allowing us to handle vector layers more efficiently, but curently we have a
        # deadline, and given how infrequently we hit this case, doing the naive thing for now is sufficient
        if (abs(self.area.left - self.area.right)) > 180.0:
            left = (-180.0 / abs_xstep) * abs_xstep
            if left < -180.0:
                left += abs_xstep
            right = (180.0 / abs_xstep) * abs_xstep
            if right > 180.0:
                right -= abs_xstep
            self.area = Area(
                left=left,
                right=right,
                top=self.area.top,
                bottom=self.area.bottom,
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

        # We have two paths: one for the common case where the hex cell doesn't cross 180˚ longitude,
        # and another case for where it does
        max_width_projection = (self.area.right - self.area.left)
        if max_width_projection < 180:

            target_window = Window(
                self.window.xoff + xoffset,
                self.window.yoff + yoffset,
                xsize,
                ysize
            )
            source_window = Window(
                xoff=0,
                yoff=0,
                xsize=self._raster_xsize,
                ysize=self._raster_ysize,
            )
            try:
                intersection = Window.find_intersection([source_window, target_window])
            except ValueError:
                return 0.0

            subset = np.zeros((intersection.ysize, intersection.xsize), dtype=float)

            start_x = self._intersection.left + ((intersection.xoff - self.window.xoff) * self._transform[1])
            start_y = self._intersection.top + ((intersection.yoff - self.window.yoff) * self._transform[5])

            for ypixel in range(intersection.ysize):

                # The latlng_to_cell is quite expensive, so we could in from either side
                # and then fill. A binary search probably would be nicer...
                left_most = intersection.xsize + 1
                right_most = -1

                lat = start_y + (ypixel * self._transform[5])

                for xpixel in range(intersection.xsize):

                    lng = start_x + (xpixel * self._transform[1])
                    this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                    if this_cell == self.cell_id:
                        # subset[ypixel][xpixel] = 1.0
                        left_most = xpixel
                        break

                for xpixel in range(intersection.xsize - 1, left_most - 1, -1):
                    lng = start_x + (xpixel * self._transform[1])
                    this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                    if this_cell == self.cell_id:
                        right_most = xpixel
                        break

                for xpixel in range(left_most, right_most + 1, 1):
                    subset[ypixel][xpixel] = 1.0

            return np.pad(
                subset,
                (
                    (
                        (intersection.yoff - self.window.yoff) - yoffset,
                        (ysize - ((intersection.yoff - self.window.yoff) + intersection.ysize)) + yoffset,
                    ),
                    (
                        (intersection.xoff - self.window.xoff) - xoffset,
                        xsize - ((intersection.xoff - self.window.xoff) + intersection.xsize) + xoffset,
                    )
                ),
                'constant'
            )
        else:
            # This handles the case where the cell wraps over 180˚ longitude
            res = np.zeros((ysize, xsize), dtype=float)

            left = min(x[1] for x in self.cell_boundary if x[1] > 0.0)
            right = max(x[1] for x in self.cell_boundary if x[1] < 0.0) + 360.0
            max_width_projection = right - left
            max_width = ceil(max_width_projection / self._transform[1])

            for ypixel in range(yoffset, yoffset + ysize):
                lat = self._intersection.top + (ypixel * self._transform[5])

                for xpixel in range(xoffset, min(xoffset + xsize, max_width)):
                    lng = self._intersection.left + (xpixel * self._transform[1])
                    this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                    if this_cell == self.cell_id:
                        res[ypixel - yoffset][xpixel - xoffset] = 1.0

                for xpixel in range(xoffset + xsize - 1, xoffset + xsize - max_width, -1):
                    lng = self._intersection.left + (xpixel * self._transform[1])
                    this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                    if this_cell == self.cell_id:
                        res[ypixel - yoffset][xpixel - xoffset] = 1.0
            return res
