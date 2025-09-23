from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy import ma

from ..rounding import round_down_pixels
from ..window import Area, Window
from .base import YirgacheffeLayer
from .rasters import RasterLayer
from .._backends import backend
from .._backends.enumeration import dtype as DataType

class GroupLayerEmpty(ValueError):
    def __init__(self, msg):
        self.msg = msg

class GroupLayer(YirgacheffeLayer):

    @classmethod
    def layer_from_directory(
        cls,
        directory_path: Path | str,
        name: str | None = None,
        matching: str = "*.tif"
    ) -> GroupLayer:
        if directory_path is None:
            raise ValueError("Directory path is None")
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)
        files = list(directory_path.glob(matching))
        if len(files) < 1:
            raise GroupLayerEmpty(directory_path)
        return cls.layer_from_files(files, name)

    @classmethod
    def layer_from_files(
        cls,
        filenames: Sequence[Path | str],
        name: str | None = None
    ) -> GroupLayer:
        if filenames is None:
            raise ValueError("filenames argument is None")
        rasters: list[YirgacheffeLayer] = [RasterLayer.layer_from_file(x) for x in filenames]
        if len(rasters) < 1:
            raise GroupLayerEmpty("No files found")
        return cls(rasters, name)

    def __init__(
        self,
        layers: list[YirgacheffeLayer],
        name: str | None = None
    ) -> None:
        if not layers:
            raise GroupLayerEmpty("Expected one or more layers")
        if not all(x.map_projection == layers[0].map_projection for x in layers):
            raise ValueError("Not all layers are the same projection/scale")
        for layer in layers:
            if layer._active_area is not None:
                raise ValueError("Layers can not currently be constrained")

        # area/window are superset of all tiles
        union = YirgacheffeLayer.find_union(layers)
        super().__init__(union, layers[0].map_projection, name=name)

        # We store them in reverse order so that from the user's perspective
        # the first layer in the list will be the most important in terms
        # over overlapping.
        self._underlying_layers = copy.copy(layers)
        self._underlying_layers.reverse()
        self.layers = self._underlying_layers

    def _park(self) -> None:
        for layer in self.layers:
            try:
                layer._park()
            except AttributeError:
                pass

    @property
    def datatype(self) -> DataType:
        return DataType.of_gdal(self.layers[0].datatype)

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

    def _read_array_with_window(
        self,
        xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:
        if (xsize <= 0) or (ysize <= 0):
            raise ValueError("Request dimensions must be positive and non-zero")

        map_projection = self.map_projection
        assert map_projection is not None
        scale = map_projection.scale

        target_window = Window(
            window.xoff + xoffset,
            window.yoff + yoffset,
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
                data = layer._read_array(
                    intersection.xoff - adjusted_layer_window.xoff,
                    intersection.yoff - adjusted_layer_window.yoff,
                    intersection.xsize,
                    intersection.ysize
                )
                if layer.nodata is not None:
                    data = backend.where(backend.isnan(data), 0.0, data)
                return data

        result = np.zeros((ysize, xsize), dtype=float)
        for layer, adjusted_layer_window, intersection in contributing_layers:
            data = layer._read_array(
                intersection.xoff - adjusted_layer_window.xoff,
                intersection.yoff - adjusted_layer_window.yoff,
                intersection.xsize,
                intersection.ysize
            )
            result_x_offset = (intersection.xoff - xoffset) - window.xoff
            result_y_offset = (intersection.yoff - yoffset) - window.yoff
            if layer.nodata is None:
                result[
                    result_y_offset:result_y_offset + intersection.ysize,
                    result_x_offset:result_x_offset + intersection.xsize
                ] = data
            else:
                masked = ma.masked_invalid(data)
                before = result[
                    result_y_offset:result_y_offset + intersection.ysize,
                    result_x_offset:result_x_offset + intersection.xsize
                ]
                merged = ma.where(masked.mask, before, masked)
                result[
                    result_y_offset:result_y_offset + intersection.ysize,
                    result_x_offset:result_x_offset + intersection.xsize
                ] = merged

        return backend.promote(result)

class TileData:
    """This class exists just to let me sort the tiles into the correct order for processing."""

    def __init__(self, data: Any, x: int, y: int):
        self.data = data
        self.x = x
        self.y = y

    @property
    def origin(self):
        return (self.x, self.y)

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

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
        if self.data is not None:
            return f"<Tile: {self.x} {self.y} {self.data.shape[1]} {self.data.shape[0]}>"
        else:
            return "<Tile: sentinal>"

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
    def _read_array_with_window(
        self,
        xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:
        if (xsize <= 0) or (ysize <= 0):
            raise ValueError("Request dimensions must be positive and non-zero")

        scale = self.pixel_scale
        assert scale is not None

        target_window = Window(
            window.xoff + xoffset,
            window.yoff + yoffset,
            xsize,
            ysize
        )

        partials: list[TileData] = []
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
            data = layer._read_array(
                intersection.xoff - adjusted_layer_window.xoff,
                intersection.yoff - adjusted_layer_window.yoff,
                intersection.xsize,
                intersection.ysize
            )
            result_x_offset = (intersection.xoff - xoffset) - window.xoff
            result_y_offset = (intersection.yoff - yoffset) - window.yoff
            partials.append(TileData(data, result_x_offset, result_y_offset))

        sorted_partials = sorted(partials)

        # When tiles overlap (hello JRC Annual Change!), then if the read aligns
        # with the edge of a tile, then we can end up with multiple results at the same
        # offset as we get the dregs of the above/left tile and the bulk of the data from
        # the "obvious" tile. In which case, we should reject the smaller section. If we have
        # two tiles at the same offset and one is not a perfect subset of the other then the
        # tile set we were given is not regularly shaped, and so we should give up.
        combed_partials: list[TileData] = []
        previous_tile = None
        for tile in sorted_partials:
            if previous_tile is None:
                previous_tile = tile
                continue

            if previous_tile.origin == tile.origin:
                if (tile.width >= previous_tile.width) and \
                    (tile.height >= previous_tile.height):
                    previous_tile = tile
                continue

            combed_partials.append(previous_tile)
            previous_tile = tile
        if previous_tile:
            combed_partials.append(previous_tile)

        # Add a terminator to force the last row to be added
        combed_partials.append(TileData(None, 0, -1))
        last_y_offset = None
        last_y_height = 0
        expected_next_x = 0
        expected_next_y = 0
        data = None
        row_chunk: np.ndarray | None = None

        # Allow for reading off top
        if combed_partials:
            if combed_partials[0].y > 0:
                row_chunk = np.zeros((combed_partials[0].y, xsize))
                last_y_offset = 0
                last_y_height = combed_partials[0].y

        for tile in combed_partials:
            if tile.y == last_y_offset:
                assert row_chunk is not None
                if row_chunk.shape[0] < tile.data.shape[0]:
                    assert last_y_height == row_chunk.shape[0]
                    row_chunk = np.vstack(
                        (row_chunk, np.zeros((tile.data.shape[0] - row_chunk.shape[0], row_chunk.shape[1])))
                    )
                    last_y_height = row_chunk.shape[0]
                new_data = tile.data
                if row_chunk.shape[0] != new_data.shape[0]:
                    assert row_chunk.shape[0] > new_data.shape[0]
                    # we have some overlap data from oversized tiles (hello JRC) when there's a GAP in general
                    new_data = np.vstack(
                        (new_data, np.zeros((row_chunk.shape[0] - new_data.shape[0], new_data.shape[1])))
                    )
                assert row_chunk.shape[0] == new_data.shape[0]

                # We're adding a tile to an existing row
                x_offset = expected_next_x - tile.x
                if x_offset == 0:
                    # Tiles line up neatly!
                    row_chunk = np.hstack((row_chunk, new_data))
                    expected_next_x = expected_next_x + new_data.shape[1]
                elif x_offset > 0:
                    # tiles overlap
                    remainder = new_data.shape[1] - xoffset
                    if remainder > 0:
                        subdata = np.delete(new_data, np.s_[0:x_offset], 1)
                        row_chunk = np.hstack((row_chunk, subdata))
                        expected_next_x = expected_next_x + subdata.shape[1]
                else:
                    # Gap between tiles, so fill it before adding new data
                    row_chunk = np.hstack((row_chunk, np.zeros((new_data.shape[0], -x_offset))))
                    row_chunk = np.hstack((row_chunk, new_data))
                    expected_next_x = expected_next_x + new_data.shape[1] + -x_offset
            else:
                # This is a new row, so we need to add the row in progress
                # and start a new one
                if row_chunk is not None:
                    if row_chunk.shape[1] != xsize:
                        assert row_chunk.shape[1] < xsize, f"row is too wide: expected {xsize}, is {row_chunk.shape[1]}"
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
                            assert last_y_offset is not None
                            diff = expected_next_y - last_y_offset
                            assert diff > 0, f"{expected_next_y} - {last_y_offset} <= 0 (aka {diff})"
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

        assert last_y_offset is not None
        if (last_y_offset + last_y_height) < ysize:
            data = np.vstack((data, np.zeros((ysize - (last_y_offset + last_y_height), xsize))))

        assert data is not None
        assert data.shape == (ysize, xsize)
        return backend.promote(data)
