from __future__ import annotations
import uuid
from typing import Any

from .. import __version__
from .._operators import LayerMathMixin
from .._datatypes import Area, MapProjection, Window
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
        self._window: Window | None = None
        self.name = name if name is not None else str(uuid.uuid4())

        if self.projection:
            width, height = self.projection.round_up_pixels(
                (self.area.right - self.area.left) / self.projection.xstep,
                (self.area.bottom - self.area.top) / self.projection.ystep,
            )
            self._window = Window(0, 0, width, height)

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
        """Returns the [`DataType`][yirgacheffe.DataType] of the pixels within a layer."""
        raise NotImplementedError("Must be overridden by subclass")

    @property
    def projection(self) -> MapProjection | None:
        """Returns the [`MapProjection`][yirgacheffe.MapProjection] (projection name and pixel size) of the layer."""
        return self._underlying_area.projection

    @property
    def area(self) -> Area:
        """Returns the geospatial [`Area`][yirgacheffe.Area] of the layer."""
        return self._underlying_area

    def _get_operation_area(
        self,
        projection: MapProjection | None = None,
        _force_union: bool = False,
        top_level: bool = False, # pylint: disable=W0613
    ) -> Area:
        self_projection = self.projection
        if self_projection is not None and projection is not None and self_projection != projection:
            raise ValueError("Calculation projection does not match layer projection")
        return self.area

    @property
    def _virtual_window(self) -> Window:
        if self._window is None:
            raise AttributeError("Layer has no window")
        return self._window

    @property
    def dimensions(self) -> tuple[int,int]:
        """Natural dimensions of the layer in pixels in width and height.

        If the layer is based on a vector that hasn't had a reference map projection applied yet then
        this will throw an attribute error.
        """
        if self._window is None:
            raise AttributeError("Layer has no dimensions")
        return (self._window.xsize, self._window.ysize)

    @property
    def nodata(self) -> None:
        return None

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
        assert self.projection is not None
        assert self.projection == target_projection

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

        xoff, yoff = self.projection.round_down_pixels(
            (target_area.left - self._underlying_area.left) / self.projection.xstep,
            (self._underlying_area.top - target_area.top) / (self.projection.ystep * -1.0),
        )
        xsize, ysize = self.projection.round_up_pixels(
            (target_area.right - target_area.left) / self.projection.xstep,
            (target_area.top - target_area.bottom) / (self.projection.ystep * -1.0),
        )

        target_window = Window(xoff, yoff, xsize, ysize)
        return self._read_array_with_window(x, y, width, height, target_window)

    def _read_array(self, x: int, y: int, width: int, height: int) -> Any:
        return self._read_array_with_window(x, y, width, height, self._virtual_window)

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
