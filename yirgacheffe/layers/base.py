from __future__ import annotations
from typing import Any, Sequence

import deprecation

from .. import __version__
from .._operators import LayerMathMixin
from ..rounding import almost_equal, round_up_pixels, round_down_pixels
from ..window import Area, MapProjection, PixelScale, Window
from .._backends import backend
from .._backends.enumeration import dtype as DataType

class YirgacheffeLayer(LayerMathMixin):
    """The common base class for the different layer types. Most still inherit from RasterLayer as deep down
    they end up as pixels, but this is a start to make other layers that don't need to rasterize not have
    to carry all that baggage."""

    def __init__(self,
        area: Area,
        projection: MapProjection | None,
        name: str | None = None
    ):
        # This is just to catch code that uses the old private API
        if projection is not None and not isinstance(projection, MapProjection):
            raise TypeError("projection value of wrong type")

        self._underlying_area = area
        self._active_area: Area | None = None
        self._projection = projection
        self._window: Window | None = None
        self.name = name

        self.reset_window()

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
        if self._projection:
            return self._projection.name
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
        if self._projection:
            return PixelScale(self._projection.xstep, self._projection.ystep)
        else:
            return None

    @property
    def map_projection(self) -> MapProjection | None:
        return self._projection

    @property
    def area(self) -> Area:
        if self._active_area is not None:
            return self._active_area
        else:
            return self._underlying_area

    def _get_operation_area(self, projection: MapProjection | None = None) -> Area:
        if self._projection is not None and projection is not None and self._projection != projection:
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
            raise ValueError("Not all layers are at the same projectin or pixel scale")

        layer_areas = [x._get_operation_area() for x in layers]
        intersection = Area(
            left=max(x.left for x in layer_areas),
            top=min(x.top for x in layer_areas),
            right=min(x.right for x in layer_areas),
            bottom=max(x.bottom for x in layer_areas)
        )
        if (intersection.left >= intersection.right) or (intersection.bottom >= intersection.top):
            raise ValueError('No intersection possible')
        return intersection

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
        return Area(
            left=min(x.left for x in layer_areas),
            top=max(x.top for x in layer_areas),
            right=max(x.right for x in layer_areas),
            bottom=min(x.bottom for x in layer_areas)
        )

    @property
    def geo_transform(self) -> tuple[float, float, float, float, float, float]:
        if self._projection is None:
            raise AttributeError("No geo transform for layers without explicit pixel scale")
        return (
            self.area.left, self._projection.xstep, 0.0,
            self.area.top, 0.0, self._projection.ystep
        )

    def check_pixel_scale(self, scale: PixelScale) -> bool:
        our_scale = self.pixel_scale
        if our_scale is None:
            raise ValueError("No check for layers without explicit pixel scale")
        return almost_equal(our_scale.xstep, scale.xstep) and \
            almost_equal(our_scale.ystep, scale.ystep)

    def set_window_for_intersection(self, new_area: Area) -> None:
        if self._projection is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        new_window = Window(
            xoff=round_down_pixels((new_area.left - self._underlying_area.left) / self._projection.xstep,
                self._projection.xstep),
            yoff=round_down_pixels((self._underlying_area.top - new_area.top) / (self._projection.ystep * -1.0),
                self._projection.ystep * -1.0),
            xsize=round_up_pixels(
                (new_area.right - new_area.left) / self._projection.xstep,
                self._projection.xstep
            ),
            ysize=round_up_pixels(
                (new_area.top - new_area.bottom) / (self._projection.ystep * -1.0),
                (self._projection.ystep * -1.0)
            ),
        )
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
        if self._projection is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        new_window = Window(
            xoff=round_down_pixels((new_area.left - self._underlying_area.left) / self._projection.xstep,
                self._projection.xstep),
            yoff=round_down_pixels((self._underlying_area.top - new_area.top) / (self._projection.ystep * -1.0),
                self._projection.ystep * -1.0),
            xsize=round_up_pixels(
                (new_area.right - new_area.left) / self._projection.xstep,
                self._projection.xstep
            ),
            ysize=round_up_pixels(
                (new_area.top - new_area.bottom) / (self._projection.ystep * -1.0),
                (self._projection.ystep * -1.0)
            ),
        )
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
        if self._projection:
            abs_xstep, abs_ystep = abs(self._projection.xstep), abs(self._projection.ystep)
            self._window = Window(
                xoff=0,
                yoff=0,
                xsize=round_up_pixels((self.area.right - self.area.left) / self._projection.xstep, abs_xstep),
                ysize=round_up_pixels((self.area.bottom - self.area.top) / self._projection.ystep, abs_ystep),
            )

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
            bottom=new_top + (new_window.ysize * scale.ystep)
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
        assert self._projection is not None
        assert self._projection == target_projection

        target_window = Window(
            xoff=round_down_pixels((target_area.left - self._underlying_area.left) / self._projection.xstep,
                self._projection.xstep),
            yoff=round_down_pixels((self._underlying_area.top - target_area.top) / (self._projection.ystep * -1.0),
                self._projection.ystep * -1.0),
            xsize=round_up_pixels(
                (target_area.right - target_area.left) / self._projection.xstep,
                self._projection.xstep
            ),
            ysize=round_up_pixels(
                (target_area.top - target_area.bottom) / (self._projection.ystep * -1.0),
                (self._projection.ystep * -1.0)
            ),
        )
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

    def latlng_for_pixel(self, x_coord: int, y_coord: int) -> tuple[float, float]:
        """Get geo coords for pixel. This is relative to the set view window."""
        if self._projection is None or "WGS 84" not in self._projection.name:
            raise NotImplementedError("Not yet supported for other projections")
        return (
            (y_coord * self._projection.ystep) + self.area.top,
            (x_coord * self._projection.xstep) + self.area.left
        )

    def pixel_for_latlng(self, lat: float, lng: float) -> tuple[int, int]:
        """Get pixel for geo coords. This is relative to the set view window.
        Result is rounded down to nearest pixel."""
        if self._projection is None or "WGS 84" not in self._projection.name:
            raise NotImplementedError("Not yet supported for other projections")
        return (
            round_down_pixels((lng - self.area.left) / self._projection.xstep, abs(self._projection.xstep)),
            round_down_pixels((lat - self.area.top) / self._projection.ystep, abs(self._projection.ystep)),
        )
