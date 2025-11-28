from __future__ import annotations
import operator
import uuid
from functools import reduce
from typing import Any, Sequence

import deprecation

from .. import __version__
from .._operators import LayerMathMixin
from .._datatypes import Area, MapProjection, PixelScale, Window
from .._backends import backend
from .._backends.enumeration import dtype as DataType

class YirgacheffeLayer(LayerMathMixin):
    """The common base class for the different layer types. Most still inherit from RasterLayer as deep down
    they end up as pixels, but this is a start to make other layers that don't need to rasterize not have
    to carry all that baggage."""

    def __init__(self,
        area: Area,
        name: str | None = None
    ):
        self._underlying_area = area
        self._active_area: Area | None = None
        self._window: Window | None = None
        self.name = name if name is not None else str(uuid.uuid4())

        self.reset_window()

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def _cse_hash(self) -> int | None:
        raise NotImplementedError("Must be overridden by subclass")

    def _park(self) -> None:
        pass

    def _unpark(self) -> None:
        pass

    @property
    def _raster_dimensions(self) -> tuple[int, int]:
        raise AttributeError("Does not have raster")

    @property
    def datatype(self) -> DataType:
        raise NotImplementedError("Must be overridden by subclass")

    @property
    @deprecation.deprecated(
        deprecated_in="1.7",
        removed_in="2.0",
        current_version=__version__,
        details="Use `map_projection` instead."
    )
    def projection(self) -> str | None:
        if self.map_projection:
            return self.map_projection.name
        else:
            return None

    @property
    @deprecation.deprecated(
        deprecated_in="1.7",
        removed_in="2.0",
        current_version=__version__,
        details="Use `map_projection` instead."
    )
    def pixel_scale(self) -> PixelScale | None:
        if self.map_projection:
            return self.map_projection.scale
        else:
            return None

    @property
    def map_projection(self) -> MapProjection | None:
        """Returns the map projection (projection name and pixel size) of the layer."""
        return self._underlying_area.projection

    @property
    def area(self) -> Area:
        """Returns the geospatial bounds of the layer."""
        if self._active_area is not None:
            return self._active_area
        else:
            return self._underlying_area

    def _get_operation_area(self, projection: MapProjection | None = None) -> Area:
        self_projection = self.map_projection
        if self_projection is not None and projection is not None and self_projection != projection:
            raise ValueError("Calculation projection does not match layer projection")
        return self.area

    @property
    def window(self) -> Window:
        if self._window is None:
            raise AttributeError("Layer has no window")
        return self._window

    @property
    def nodata(self) -> None:
        return None

    @staticmethod
    def find_intersection(layers: Sequence[YirgacheffeLayer]) -> Area:
        if not layers:
            raise ValueError("Expected list of layers")

        # This only makes sense (currently) if all layers
        # have the same pixel pitch (modulo desired accuracy)
        projections = [x.map_projection for x in layers if x.map_projection is not None]
        if not projections:
            raise ValueError("No layers have a projection")
        if not all(projections[0] == x for x in projections[1:]):
            raise ValueError("Not all layers are at the same projection or pixel scale")

        layer_areas = [x._get_operation_area() for x in layers]
        return reduce(operator.and_, layer_areas)

    @staticmethod
    def find_union(layers: Sequence[YirgacheffeLayer]) -> Area:
        if not layers:
            raise ValueError("Expected list of layers")

        # This only makes sense (currently) if all layers
        # have the same pixel pitch (modulo desired accuracy)
        projections = [x.map_projection for x in layers if x.map_projection is not None]
        if not projections:
            raise ValueError("No layers have a projection")
        if not all(projections[0] == x for x in projections[1:]):
            raise ValueError("Not all layers are at the same projectin or pixel scale")

        layer_areas = [x._get_operation_area() for x in layers]
        # This removal of global layers is to stop constant layers forcing everything to be global
        return reduce(operator.or_, [x for x in layer_areas if not x.is_world])

    @property
    def geo_transform(self) -> tuple[float, float, float, float, float, float]:
        if self.map_projection is None:
            raise AttributeError("No geo transform for layers without explicit pixel scale")
        return (
            self.area.left, self.map_projection.xstep, 0.0,
            self.area.top, 0.0, self.map_projection.ystep
        )

    def set_window_for_intersection(self, new_area: Area) -> None:
        if self.map_projection is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        if new_area.projection is None:
            new_area = new_area.project_like(self._underlying_area)

        # We force everything onto a grid aligned basis for calculating the window to avoid rounding issues
        new_area = new_area._grid_aligned
        underlying_area = self._underlying_area._grid_aligned

        xoff, yoff = self.map_projection.round_down_pixels(
            (new_area.left - underlying_area.left) / abs(self.map_projection.xstep),
            (underlying_area.top - new_area.top) / abs(self.map_projection.ystep),
        )
        xsize, ysize = self.map_projection.round_up_pixels(
            (new_area.right - new_area.left) / abs(self.map_projection.xstep),
            (new_area.top - new_area.bottom) / abs(self.map_projection.ystep),
        )
        new_window = Window(xoff, yoff, xsize, ysize)

        if (new_window.xoff < 0) or (new_window.yoff < 0):
            raise ValueError('Window has negative offset')
        # If there is an underlying raster for this layer, do a sanity check
        try:
            raster_xsize, raster_ysize = self._raster_dimensions
            if ((new_window.xoff + new_window.xsize) > raster_xsize) or \
                ((new_window.yoff + new_window.ysize) > raster_ysize):
                raise ValueError(f'Window is bigger than dataset: raster is {raster_xsize}x{raster_ysize}'\
                    f', new window is {new_window.xsize - new_window.xoff}x{new_window.ysize - new_window.yoff}')
        except AttributeError:
            pass
        self._window = new_window
        self._active_area = new_area

    def set_window_for_union(self, new_area: Area) -> None:
        if self.map_projection is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        if new_area.projection is None:
            new_area = new_area.project_like(self._underlying_area)

        # We force everything onto a grid aligned basis for calculating the window to avoid rounding issues
        new_area = new_area._grid_aligned
        underlying_area = self._underlying_area._grid_aligned

        xoff, yoff = self.map_projection.round_down_pixels(
            (new_area.left - underlying_area.left) / abs(self.map_projection.xstep),
            (underlying_area.top - new_area.top) / abs(self.map_projection.ystep),
        )
        xsize, ysize = self.map_projection.round_up_pixels(
            (new_area.right - new_area.left) / abs(self.map_projection.xstep),
            (new_area.top - new_area.bottom) / abs(self.map_projection.ystep),
        )
        new_window = Window(xoff, yoff, xsize, ysize)

        if (new_window.xoff > 0) or (new_window.yoff > 0):
            raise ValueError('Window has positive offset')
        # If there is an underlying raster for this layer, do a sanity check
        try:
            raster_xsize, raster_ysize = self._raster_dimensions
            if ((new_window.xsize - new_window.xoff) < raster_xsize) or \
                ((new_window.ysize - new_window.yoff) <raster_ysize):
                raise ValueError(f'Window is smaller than dataset: raster is {raster_xsize}x{raster_ysize}'\
                    f', new window is {new_window.xsize - new_window.xoff}x{new_window.ysize - new_window.yoff}')
        except AttributeError:
            pass
        self._window = new_window
        self._active_area = new_area

    def reset_window(self) -> None:
        self._active_area = None
        if self.map_projection:
            width, height = self.map_projection.round_up_pixels(
                (self.area.right - self.area.left) / self.map_projection.xstep,
                (self.area.bottom - self.area.top) / self.map_projection.ystep,
            )
            self._window = Window(0, 0, width, height)

    def offset_window_by_pixels(self, offset: int) -> None:
        """Grows (if pixels is positive) or shrinks (if pixels is negative) the window for the layer."""
        if offset == 0:
            return

        if offset < 0:
            if (offset * -2 >= self.window.xsize) or (offset * -2 >= self.window.ysize):
                raise ValueError(f"Can not shrink window by {offset}, would make size 0 or fewer pixels.")

        new_window = Window(
            xoff=self.window.xoff - offset,
            yoff=self.window.yoff - offset,
            xsize=self.window.xsize + (2 * offset),
            ysize=self.window.ysize + (2 * offset),
        )
        scale = self.pixel_scale
        if scale is None:
            raise ValueError("Can not offset Window without explicit pixel scale")

        # Note we can't assume that we weren't already on an intersection when making the offset!
        # But remember that window is always relative to underlying area, and new_window
        # here is based off the existing window
        new_left = self._underlying_area.left + (new_window.xoff * scale.xstep)
        new_top = self._underlying_area.top + (new_window.yoff * scale.ystep)
        new_area = Area(
            left=new_left,
            top=new_top,
            right=new_left + (new_window.xsize * scale.xstep),
            bottom=new_top + (new_window.ysize * scale.ystep),
            projection=self._underlying_area.projection,
        )
        self._window = new_window
        self._active_area = new_area

    def _read_array_with_window(
        self,
        _x: int,
        _y: int,
        _xsize: int,
        _ysize: int,
        _window: Window,
    ) -> Any:
        raise NotImplementedError("Must be overridden by subclass")

    def _read_array_for_area(
        self,
        target_area: Area,
        target_projection: MapProjection,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Any:
        assert self.map_projection is not None
        assert self.map_projection == target_projection

        # move the target area to align with our grid offset
        target_offset = target_area._grid_offset
        if target_offset is not None:
            self_offset = self._underlying_area._grid_offset
            assert self_offset is not None
            target_offset_x, target_offset_y = target_offset
            self_offset_x, self_offset_y = self_offset
            diff_x = self_offset_x - target_offset_x
            diff_y = self_offset_y - target_offset_y
            target_area = Area(
                target_area.left + diff_x,
                target_area.top + diff_y,
                target_area.right + diff_x,
                target_area.bottom + diff_y,
                target_area.projection,
            )

        xoff, yoff = self.map_projection.round_down_pixels(
            (target_area.left - self._underlying_area.left) / self.map_projection.xstep,
            (self._underlying_area.top - target_area.top) / (self.map_projection.ystep * -1.0),
        )
        xsize, ysize = self.map_projection.round_up_pixels(
            (target_area.right - target_area.left) / self.map_projection.xstep,
            (target_area.top - target_area.bottom) / (self.map_projection.ystep * -1.0),
        )

        target_window = Window(xoff, yoff, xsize, ysize)
        return self._read_array_with_window(x, y, width, height, target_window)

    def _read_array(self, x: int, y: int, width: int, height: int) -> Any:
        return self._read_array_with_window(x, y, width, height, self.window)

    def read_array(self, x: int, y: int, width: int, height: int) -> Any:
        """Reads data from the layer based on the current reference window.

        Args:
            x: X axis offset for reading
            y: Y axis offset for reading
            width: Width of data to read
            height: Height of data to read

        Returns:
            An array of values from the layer.
        """
        res = self._read_array(x, y, width, height)
        return backend.demote_array(res)
