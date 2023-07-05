from typing import Any, List

import numpy as np

from ..rounding import are_pixel_scales_equal_enough, round_down_pixels
from ..window import Window
from .base import YirgacheffeLayer

class GroupLayer(YirgacheffeLayer):

    def __init__(self, layers: List[YirgacheffeLayer]):
        if not layers:
            raise ValueError("Expected one or more layers")
        if not are_pixel_scales_equal_enough([x.pixel_scale for x in layers]):
            raise ValueError("Not all layers are at the same pixel scale")
        if not all(x.projection == layers[0].projection for x in layers):
            raise ValueError("Not all layers are the same projection")
        for layer in layers:
            if layer._active_area != layer._underlying_area:
                raise ValueError("Layers can not currently be constrained")

        # area/window are superset of all tiles
        union = YirgacheffeLayer.find_union(layers)
        super().__init__(union, layers[0].pixel_scale, layers[0].projection)

        self.layers = layers

    def read_array(self, xoffset: int, yoffset: int, xsize: int, ysize: int) -> Any:
        # Do a naive implementation to start with, and we can improve on
        # this over time
        result = np.zeros((ysize, xsize), dtype=float)

        scale = self.pixel_scale
        assert scale is not None

        target_window = Window(
            self.window.xoff + xoffset,
            self.window.yoff + yoffset,
            xsize,
            ysize
        )

        for layer in self.layers:
            # Normally this is hidden with set_window_for_...
            adjusted_layer_window = Window(
                layer.window.xoff + \
                    round_down_pixels(((layer.area.left - self._underlying_area.left) / scale.xstep), abs(scale.xstep)),
                layer.window.yoff + \
                    round_down_pixels(((layer.area.top - self._underlying_area.top) / scale.ystep), abs(scale.ystep)),
                layer.window.xsize,
                layer.window.ysize,
            )
            try:
                intersection = Window.find_intersection([target_window, adjusted_layer_window])
                data = layer.read_array(
                    intersection.xoff - adjusted_layer_window.xoff,
                    intersection.yoff - adjusted_layer_window.yoff,
                    intersection.xsize,
                    intersection.ysize
                )
                result_x_offset = (intersection.xoff - xoffset) - self.window.xoff
                result_y_offset = (intersection.yoff - yoffset) - self.window.yoff
                result[
                    result_y_offset:result_y_offset + intersection.ysize,
                    result_x_offset:result_x_offset + intersection.xsize
                ] = data
            except ValueError:
                continue

        return result
