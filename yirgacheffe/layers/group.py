import copy
from typing import Any, List, Optional

import numpy as np

from ..rounding import are_pixel_scales_equal_enough, round_down_pixels
from ..window import Area, Window
from .base import YirgacheffeLayer

class GroupLayer(YirgacheffeLayer):

    def __init__(self, layers: List[YirgacheffeLayer], name: Optional[str] = None):
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
        super().__init__(union, layers[0].pixel_scale, layers[0].projection, name=name)

        # We store them in reverse order so that from the user's perspective
        # the first layer in the list will be the most important in terms
        # over overlapping.
        self._underlying_layers = copy.copy(layers)
        self._underlying_layers.reverse()
        self.layers = self._underlying_layers

    def set_window_for_intersection(self, new_area: Area) -> None:
        super().set_window_for_intersection(new_area)

        # filter out layers we don't care about
        self.layers = [layer for layer in self._underlying_layers if layer.area.overlaps(new_area)]

    def set_window_for_union(self, new_area: Area) -> None:
        super().set_window_for_union(new_area)

        # filter out layers we don't care about
        self.layers = [layer for layer in self._underlying_layers if layer.area.overlaps(new_area)]

    def reset_window(self) -> None:
        super().reset_window()
        try:
            self.layers = self._underlying_layers
        except AttributeError:
            pass # called from Base constructor before we've added the extra field

    def read_array(self, xoffset: int, yoffset: int, xsize: int, ysize: int) -> Any:
        scale = self.pixel_scale
        assert scale is not None

        target_window = Window(
            self.window.xoff + xoffset,
            self.window.yoff + yoffset,
            xsize,
            ysize
        )

        contributing_layers = []
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
            intersection = Window.find_intersection_no_throw([target_window, adjusted_layer_window])
            if intersection is not None:
                contributing_layers.append((layer, adjusted_layer_window, intersection))

        # Adding the numpy arrays over each other is relatively expensive, so if we only intersect
        # with a single layer, turn this into a direct read
        if len(contributing_layers) == 1:
            layer, adjusted_layer_window, intersection = contributing_layers[0]
            if target_window == intersection:
                return layer.read_array(
                    intersection.xoff - adjusted_layer_window.xoff,
                    intersection.yoff - adjusted_layer_window.yoff,
                    intersection.xsize,
                    intersection.ysize
                )

        result = np.zeros((ysize, xsize), dtype=float)
        for layer, adjusted_layer_window, intersection in contributing_layers:
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

        return result

class TileData:
    """This class exists just to let me sort the tiles into the correct order for processing."""

    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y

    def __cmp__(self, other):
        if self.y != other.y:
            return self.y.cmp(other.y)
        else:
            return self.x.cmp(other.x)

    def __gt__(self, other):
        if self.y > other.y:
            return True
        if self.y < other.y:
            return False
        return self.x.__gt__(other.x)

    def __repr__(self):
        return f"<Tile: {self.x} {self.y} {self.data.shape[1]} {self.data.shape[0]}>"

class TiledGroupLayer(GroupLayer):
    """An optimised version of GroupLayer for the case where you have a grid of regular sized
    layers, e.g., map tiles.

    This class exists as assembling arbitrary shaped tiles into a group is quite slow due
    to numpy being slow to modify one array with the contents of another. This class does
    away with that at the expense of needing to know all tiles have the same shape.__abs__()

    Two notes:
    * You can have missing tiles, and it'll fill in zeros.
    * The tiles can overlap - e.g., JRC Annual Change tiles all overlap by a few pixels on all edges.
    """

    def read_array(self, xoffset: int, yoffset: int, xsize: int, ysize: int) -> Any:
        scale = self.pixel_scale
        assert scale is not None

        target_window = Window(
            self.window.xoff + xoffset,
            self.window.yoff + yoffset,
            xsize,
            ysize
        )

        partials = []
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
            intersection = Window.find_intersection_no_throw([target_window, adjusted_layer_window])
            if intersection is None:
                continue
            data = layer.read_array(
                intersection.xoff - adjusted_layer_window.xoff,
                intersection.yoff - adjusted_layer_window.yoff,
                intersection.xsize,
                intersection.ysize
            )
            result_x_offset = (intersection.xoff - xoffset) - self.window.xoff
            result_y_offset = (intersection.yoff - yoffset) - self.window.yoff
            partials.append(TileData(data, result_x_offset, result_y_offset))

        sorted_partials = sorted(partials)
        # Add a terminator to force the last row to be added
        sorted_partials.append(TileData(None, 0, -1))
        last_y_offset = None
        last_y_height = 0
        expected_next_x = 0
        expected_next_y = 0
        data = None
        row_chunk = None
        for tile in sorted_partials:
            if tile.y == last_y_offset:
                assert row_chunk is not None
                # We're adding a tile to an existing row
                x_offset = expected_next_x - tile.x
                if x_offset == 0:
                    # Tiles line up neatly!
                    row_chunk = np.hstack((row_chunk, tile.data))
                    expected_next_x = expected_next_x + tile.data.shape[1]
                elif x_offset > 0:
                    # tiles overlap
                    subdata = np.delete(tile.data, np.s_[0:x_offset], 1)
                    row_chunk = np.hstack((row_chunk, subdata))
                    expected_next_x = expected_next_x + subdata.shape[1]
                else:
                    # Gap between tiles, so fill it before adding new data
                    row_chunk = np.hstack((row_chunk, np.zeros((tile.data.shape[0], -x_offset))))
                    row_chunk = np.hstack((row_chunk, tile.data))
                    expected_next_x = expected_next_x + tile.data.shape[1] + x_offset
            else:
                # This is a new row, so we need to add the row in progress
                # and start a new one
                if row_chunk is not None:
                    if row_chunk.shape[1] != xsize:
                        # Missing tile at end of row, so fill in
                        row_chunk = np.hstack((row_chunk, np.zeros((last_y_height, xsize - row_chunk.shape[1]))))
                    if data is None:
                        data = row_chunk
                        expected_next_y += last_y_height
                    else:
                        if last_y_offset == expected_next_y:
                            data = np.vstack((data, row_chunk))
                            expected_next_y += last_y_height
                        else:
                            diff = expected_next_y - last_y_offset
                            assert diff > 0
                            subdata = np.delete(row_chunk, np.s_[0:diff], 0)
                            data = np.vstack((data, subdata))
                            expected_next_y += subdata.shape[0]
                if tile.data is not None:
                    if tile.x != 0:
                        row_chunk = np.hstack((np.zeros((tile.data.shape[0], tile.x)), tile.data))
                    else:
                        row_chunk = tile.data
                    last_y_offset = tile.y
                    last_y_height = tile.data.shape[0]
                    expected_next_x = tile.data.shape[1] + tile.x

        assert data.shape == (ysize, xsize)
        return data
