
from typing import List, Optional, Tuple

from ..operators import LayerMathMixin
from ..rounding import almost_equal, are_pixel_scales_equal_enough, round_up_pixels, round_down_pixels
from ..window import Area, PixelScale, Window

class YirgacheffeLayer(LayerMathMixin):
    """The common base class for the different layer types. Most still inherit from RasterLayer as deep down
    they end up as pixels, but this is a start to make other layers that don't need to rasterize not have
    to carry all that baggage."""

    def __init__(self, area: Area, pixel_scale: Optional[PixelScale], projection: str):
        self._pixel_scale = pixel_scale
        self._area = area
        self._intersection = area
        self._projection = projection

    @property
    def projection(self) -> str:
        return self._projection

    @property
    def pixel_scale(self) -> Optional[PixelScale]:
        return self._pixel_scale

    @property
    def area(self) -> Area:
        return self._area

    @staticmethod
    def find_intersection(layers: List) -> Area:
        if not layers:
            raise ValueError("Expected list of layers")

        # This only makes sense (currently) if all layers
        # have the same pixel pitch (modulo desired accuracy)
        if not are_pixel_scales_equal_enough([x.pixel_scale for x in layers]):
            raise ValueError("Not all layers are at the same pixel scale")

        intersection = Area(
            left=max(x.area.left for x in layers),
            top=min(x.area.top for x in layers),
            right=min(x.area.right for x in layers),
            bottom=max(x.area.bottom for x in layers)
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
            left=min(x.area.left for x in layers),
            top=max(x.area.top for x in layers),
            right=max(x.area.right for x in layers),
            bottom=min(x.area.bottom for x in layers)
        )

    @property
    def geo_transform(self) -> Tuple[float, float, float, float, float, float]:
        if self._pixel_scale is None:
            raise ValueError("No geo transform for layers without explicit pixel scale")
        if self._intersection:
            return (
                self._intersection.left, self._pixel_scale.xstep,
                0.0, self._intersection.top, 0.0, self._pixel_scale.ystep
            )
        return (
            self._area.left, self._pixel_scale.xstep, 0.0,
            self._area.top, 0.0, self._pixel_scale.ystep
        )

    def check_pixel_scale(self, scale: PixelScale) -> bool:
        our_scale = self.pixel_scale
        if our_scale is None:
            raise ValueError("No check for layers without explicit pixel scale")
        return almost_equal(our_scale.xstep, scale.xstep) and \
            almost_equal(our_scale.ystep, scale.ystep)

    def set_window_for_intersection(self, intersection: Area) -> None:
        if self._pixel_scale is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        new_window = Window(
            xoff=round_down_pixels((intersection.left - self.area.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep),
            yoff=round_down_pixels((self.area.top - intersection.top) / (self._pixel_scale.ystep * -1.0),
                self._pixel_scale.ystep * -1.0),
            xsize=round_up_pixels(
                (intersection.right - intersection.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep
            ),
            ysize=round_up_pixels(
                (intersection.top - intersection.bottom) / (self._pixel_scale.ystep * -1.0),
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
        self.window = new_window
        self._intersection = intersection

    def set_window_for_union(self, intersection: Area) -> None:
        if self._pixel_scale is None:
            raise ValueError("Can not set Window without explicit pixel scale")

        new_window = Window(
            xoff=round_down_pixels((intersection.left - self.area.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep),
            yoff=round_down_pixels((self.area.top - intersection.top) / (self._pixel_scale.ystep * -1.0),
                self._pixel_scale.ystep * -1.0),
            xsize=round_up_pixels(
                (intersection.right - intersection.left) / self._pixel_scale.xstep,
                self._pixel_scale.xstep
            ),
            ysize=round_up_pixels(
                (intersection.top - intersection.bottom) / (self._pixel_scale.ystep * -1.0),
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
        self.window = new_window
        self._intersection = intersection

    def reset_window(self):
        self._intersection = None
        self.window = Window(
            xoff=0,
            yoff=0,
            xsize=self._dataset.RasterXSize,
            ysize=self._dataset.RasterYSize,
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
        # we store the new intersection to be consistent with the set_window_for_... methods
        # Note we can't assume that we weren't already on an intersection when making the offset!
        scale = self.pixel_scale
        if scale is None:
            raise ValueError("Can not offset Window without explicit pixel scale")

        new_left = self.area.left + (new_window.xoff * scale.xstep)
        new_top = self.area.top + (new_window.yoff * scale.ystep)
        intersection = Area(
            left=new_left,
            top=new_top,
            right=new_left + (new_window.xsize * scale.xstep),
            bottom=new_top + (new_window.ysize * scale.ystep)
        )
        self.window = new_window
        self._intersection = intersection
