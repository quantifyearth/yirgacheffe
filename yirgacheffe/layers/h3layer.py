from __future__ import annotations

from math import ceil, floor
from typing import Any

import h3
import numpy as np

from ..rounding import round_up_pixels
from ..window import Area, MapProjection, Window
from .base import YirgacheffeLayer
from .._backends import backend
from .._backends.enumeration import dtype as DataType

class H3CellLayer(YirgacheffeLayer):

    def __init__(self, cell_id: str, projection: MapProjection):
        if not h3.is_valid_cell(cell_id):
            raise ValueError(f"{cell_id} is not a valid H3 cell identifier")
        self.cell_id = cell_id
        self.zoom = h3.get_resolution(cell_id)

        self.cell_boundary = h3.cell_to_boundary(cell_id)

        abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)
        area = Area(
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
        if not area.bottom < self.centre[0] < area.top:
            raise NotImplementedError("Distortion at poles not currently handled")

        # For wrap around due to hitting the longitunal wrap, we currently just do the naive thing and
        # project a band for the full width of the projection. I have a plan to fix this, as I'd like layers
        # to have sublayers, allowing us to handle vector layers more efficiently, but curently we have a
        # deadline, and given how infrequently we hit this case, doing the naive thing for now is sufficient
        if (abs(area.left - area.right)) > 180.0:
            left = (-180.0 / abs_xstep) * abs_xstep
            if left < -180.0:
                left += abs_xstep
            right = (180.0 / abs_xstep) * abs_xstep
            if right > 180.0:
                right -= abs_xstep
            area = Area(
                left=left,
                right=right,
                top=area.top,
                bottom=area.bottom,
            )

        super().__init__(area, projection)
        self._window = Window(
            xoff=0,
            yoff=0,
            xsize=round_up_pixels((area.right - area.left) / projection.xstep, abs_xstep),
            ysize=round_up_pixels((area.bottom - area.top) / projection.ystep, abs_ystep),
        )
        self._raster_xsize = self.window.xsize
        self._raster_ysize = self.window.ysize

        # see comment in read_array for why we're doing this
        sorted_lats = [x[0] for x in self.cell_boundary]
        sorted_lats.sort()
        self._raster_safe_bounds = (
            (sorted_lats[-2] / abs_ystep) * abs_ystep,
            (sorted_lats[1] / abs_ystep) * abs_ystep,
        )

    @property
    def _raster_dimensions(self) -> tuple[int, int]:
        return (self._raster_xsize, self._raster_ysize)

    @property
    def datatype(self) -> DataType:
        return DataType.Float64

    def _read_array_with_window(
        self,
        xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:
        assert self._projection is not None

        if (xsize <= 0) or (ysize <= 0):
            raise ValueError("Request dimensions must be positive and non-zero")

        # We have two paths: one for the common case where the hex cell doesn't cross 180˚ longitude,
        # and another case for where it does
        max_width_projection = self.area.right - self.area.left
        if max_width_projection < 180:

            target_window = Window(
                window.xoff + xoffset,
                window.yoff + yoffset,
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

            subset = np.zeros((intersection.ysize, intersection.xsize))

            start_x = self.area.left + ((intersection.xoff - self.window.xoff) * self._projection.xstep)
            start_y = self.area.top + ((intersection.yoff - self.window.yoff) * self._projection.ystep)

            for ypixel in range(intersection.ysize):
                # The latlng_to_cell is quite expensive, so ideally we want to avoid
                # calling latlng_to_cell for every pixel, though I do want to do that
                # rather than infer from the cell_boundary, as this way I'm ensured
                # I get an authoritative result and no pixel falls between the cracks.

                # Originally I just tried the old-school way of finding the left-most
                # and the right-most pixel per row, and then filling between, which in
                # theory should work fine for rasterising hexagons, but in certain places
                # the map projection distorts the edges sufficiently they become concave,
                # and that approach lead to over filling, which caused issues with cells
                # overlapping.

                # The current implementation tries to hedge its bets by looking where the
                # lowest and highest edges are (stored in _raster_safe_bounds) and then
                # only doing the fill between left and right within those bounds. I suspect
                # longer term this will need some padding, but it works for the current set
                # of known failures in test_optimisation. As we hit more issues we should
                # expand that test case before tweaking here. This is quite wasteful, as it's
                # only a tiny number of cells that have convex edges, so in future it'd be
                # interesting to see if we can infer when that is.

                lat = start_y + (ypixel * self._projection.ystep)

                if self._raster_safe_bounds[0] < lat < self._raster_safe_bounds[1]:
                    # In "safe" zone, try to be clever
                    left_most = intersection.xsize + 1
                    right_most = -1

                    for xpixel in range(intersection.xsize):
                        lng = start_x + (xpixel * self._projection.xstep)
                        this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                        if this_cell == self.cell_id:
                            left_most = xpixel
                            break

                    for xpixel in range(intersection.xsize - 1, left_most - 1, -1):
                        lng = start_x + (xpixel * self._projection.xstep)
                        this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                        if this_cell == self.cell_id:
                            right_most = xpixel
                            break

                    for xpixel in range(left_most, right_most + 1, 1):
                        subset[ypixel][xpixel] = 1.0
                else:
                    # Not in safe zone, be diligent.
                    for xpixel in range(intersection.xsize):
                        lng = start_x + (xpixel * self._projection.xstep)
                        this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                        if this_cell == self.cell_id:
                            subset[ypixel][xpixel] = 1.0

            return backend.pad(
                backend.promote(subset),
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
            res = np.zeros((ysize, xsize))

            left = min(x[1] for x in self.cell_boundary if x[1] > 0.0)
            right = max(x[1] for x in self.cell_boundary if x[1] < 0.0) + 360.0
            max_width_projection = right - left
            max_width = ceil(max_width_projection / self._projection.xstep)

            for ypixel in range(yoffset, yoffset + ysize):
                lat = self.area.top + (ypixel * self._projection.ystep)

                for xpixel in range(xoffset, min(xoffset + xsize, max_width)):
                    lng = self.area.left + (xpixel * self._projection.xstep)
                    this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                    if this_cell == self.cell_id:
                        res[ypixel - yoffset][xpixel - xoffset] = 1.0

                for xpixel in range(xoffset + xsize - 1, xoffset + xsize - max_width, -1):
                    lng = self.area.left + (xpixel * self._projection.xstep)
                    this_cell = h3.latlng_to_cell(lat, lng, self.zoom)
                    if this_cell == self.cell_id:
                        res[ypixel - yoffset][xpixel - xoffset] = 1.0
            return backend.promote(res)
