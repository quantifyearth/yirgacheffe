
from typing import Any, List, Optional, Tuple

from ..operators import LayerMathMixin
from ..rounding import almost_equal, are_pixel_scales_equal_enough, round_up_pixels, round_down_pixels
from ..window import Area, PixelScale, Window

class YirgacheffeLayer(LayerMathMixin):
    """The common base class for the different layer types. Most still inherit from RasterLayer as deep down
    they end up as pixels, but this is a start to make other layers that don't need to rasterize not have
    to carry all that baggage."""

    def __init__(self,
        area: Area,
        pixel_scale: Optional[PixelScale],
        projection: str
    ):
        self._pixel_scale = pixel_scale
        self._underlying_area = area
        self._active_area = area
        self._projection = projection

        self.reset_window()

    @property
    def projection(self) -> str:
        return self._projection

    @property
    def pixel_scale(self) -> Optional[PixelScale]:
        return self._pixel_scale

    @property
    def area(self) -> Area:
        return self._active_area

    @property
    def window(self) -> Window:
        return self._window

    @staticmethod
    def find_intersection(layers: List) -> Area:
        if not layers:
            raise ValueError("Expected list of layers")

        # This only makes sense (currently) if all layers
        # have the same pixel pitch (modulo desired accuracy)
        if not are_pixel_scales_equal_enough([x.pixel_scale for x in layers]):
            raise ValueError("Not all layers are at the same pixel scale")

        intersection = Area(
            left=max(x._underlying_area.left for x in layers),
            top=min(x._underlying_area.top for x in layers),
            right=min(x._underlying_area.right for x in layers),
            bottom=max(x._underlying_area.bottom for x in layers)
        )
        if (intersection.left >= intersection.right) or (intersection.bottom >= intersection.top):
            raise ValueError('No intersection possible')
        return intersection

    @staticmethod
    def find_union(layers: List) -> Area:
        if not layers:
            raise ValueError("Expected list of layers")

        # This only makes sense (currently) if all layers
        # have the same pixel pitch (modulo desired accuracy)
        if not are_pixel_scales_equal_enough([x.pixel_scale for x in layers]):
            raise ValueError("Not all layers are at the same pixel scale")

        return Area(
            left=min(x._underlying_area.left for x in layers),
            top=max(x._underlying_area.top for x in layers),
            right=max(x._underlying_area.right for x in layers),
            bottom=min(x._underlying_area.bottom for x in layers)
        )

    @property
    def geo_transform(self) -> Tuple[float, float, float, float, float, float]:
        if self._pixel_scale is None:
            raise ValueError("No geo transform for layers without explicit pixel scale")
        return (
            self._active_area.left, self._pixel_scale.xstep,
            0.0, self._active_area.top, 0.0, self._pixel_scale.ystep
        )

    def check_pixel_scale(self, scale: PixelScale) -> bool:
        our_scale = self.pixel_scale
        if our_scale is None:
            raise ValueError("No check for layers without explicit pixel scale")
        return almost_equal(our_scale.xstep, scale.xstep) and \
            almost_equal(our_scale.ystep, scale.ystep)

    def set_window_for_intersection(self, new_area: Area) -> None:
        if self._pixel_scale is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        new_window = Window(
            xoff=round_down_pixels((new_area.left - self._underlying_area.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep),
            yoff=round_down_pixels((self._underlying_area.top - new_area.top) / (self._pixel_scale.ystep * -1.0),
                self._pixel_scale.ystep * -1.0),
            xsize=round_up_pixels(
                (new_area.right - new_area.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep
            ),
            ysize=round_up_pixels(
                (new_area.top - new_area.bottom) / (self._pixel_scale.ystep * -1.0),
                (self._pixel_scale.ystep * -1.0)
            ),
        )
        if (new_window.xoff < 0) or (new_window.yoff < 0):
            raise ValueError('Window has negative offset')
        # If there is an underlying raster for this layer, do a sanity check
        try:
            if ((new_window.xoff + new_window.xsize) > self._raster_xsize) or \
                ((new_window.yoff + new_window.ysize) > self._raster_ysize):
                raise ValueError(f'Window is bigger than dataset: raster is {self._raster_xsize}x{self._raster_ysize}'\
                    f', new window is {new_window.xsize - new_window.xoff}x{new_window.ysize - new_window.yoff}')
        except AttributeError:
            pass
        self._window = new_window
        self._active_area = new_area

    def set_window_for_union(self, new_area: Area) -> None:
        if self._pixel_scale is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        new_window = Window(
            xoff=round_down_pixels((new_area.left - self._underlying_area.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep),
            yoff=round_down_pixels((self._underlying_area.top - new_area.top) / (self._pixel_scale.ystep * -1.0),
                self._pixel_scale.ystep * -1.0),
            xsize=round_up_pixels(
                (new_area.right - new_area.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep
            ),
            ysize=round_up_pixels(
                (new_area.top - new_area.bottom) / (self._pixel_scale.ystep * -1.0),
                (self._pixel_scale.ystep * -1.0)
            ),
        )
        if (new_window.xoff > 0) or (new_window.yoff > 0):
            raise ValueError('Window has positive offset')
        # If there is an underlying raster for this layer, do a sanity check
        try:
            if ((new_window.xsize - new_window.xoff) < self._raster_xsize) or \
                ((new_window.ysize - new_window.yoff) < self._raster_ysize):
                raise ValueError(f'Window is smaller than dataset: raster is {self._raster_xsize}x{self._raster_ysize}'\
                    f', new window is {new_window.xsize - new_window.xoff}x{new_window.ysize - new_window.yoff}')
        except AttributeError:
            pass
        self._window = new_window
        self._active_area = new_area

    def reset_window(self):
        self._active_area = self._underlying_area
        if self._pixel_scale:
            abs_xstep, abs_ystep = abs(self._pixel_scale.xstep), abs(self._pixel_scale.ystep)
            self._window = Window(
                xoff=0,
                yoff=0,
                xsize=round_up_pixels((self.area.right - self.area.left) / self._pixel_scale.xstep, abs_xstep),
                ysize=round_up_pixels((self.area.bottom - self.area.top) / self._pixel_scale.ystep, abs_ystep),
            )

    def offset_window_by_pixels(self, offset: int):
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

    def read_array(self, _x: int, _y: int, _xsize: int, _ysize: int) -> Any:
        raise NotImplementedError("Must be overridden by subclass")
