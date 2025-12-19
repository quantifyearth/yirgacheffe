from __future__ import annotations
import os
from typing import Any

from osgeo import gdal

from .._datatypes import Area, MapProjection, Window
from .._core import read_raster
from .rasters import YirgacheffeLayer, RasterLayer
from .._backends.enumeration import dtype as DataType

class VsimemFile:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, *args):
        try:
            gdal.Unlink(self.path)
        except RuntimeError:
            pass

class ReprojectedRasterLayer(YirgacheffeLayer):
    """ReprojectedRasterLayer dynamically reprojects a layer."""

    def __init__(
        self,
        src: RasterLayer,
        target_projection: MapProjection,
        method: str = "nearest",
        name: str | None = None,
    ):
        reprojected_area = src.area.reproject(target_projection)

        super().__init__(
            reprojected_area,
            name=name
        )

        self._src = src
        self._method = method

    @property
    def _cse_hash(self) -> int | None:
        return hash((
            self._src._cse_hash,
            self.name,
            self._underlying_area,
            self._method,
            self.map_projection,
            self._active_area
        ))

    def close(self):
        self._src.close()

    def _park(self):
        self._src._park()

    def _unpark(self):
        self._src._unpark()

    @property
    def datatype(self) -> DataType:
        return self._src.datatype

    def _read_array_with_window(
        self,
        xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:

        # Step 1: work out what the area trying to be read is
        xoffset = xoffset + window.xoff
        yoffset = yoffset + window.yoff

        underlying_area = self._underlying_area
        projection = underlying_area.projection
        assert projection is not None
        read_area = Area(
            left=underlying_area.left + (xoffset * projection.xstep),
            top=underlying_area.top + (yoffset * projection.ystep),
            right=underlying_area.left + (xoffset * projection.xstep) + (xsize * projection.xstep),
            bottom=underlying_area.top + (yoffset * projection.ystep) + (ysize * projection.ystep),
            projection=projection,
        )

        expand_buffer = 1 # This should probably be some variable based on the method and direction?

        # now we want this area in the source projection
        src_projection = self._src.map_projection
        assert src_projection is not None
        src_read_area = read_area.reproject(src_projection)
        expanded_src_read_area = src_read_area.grow(expand_buffer * src_projection.xstep)

        # We need some ID that stops us with other parallel workers potentially in the
        # VSIMEM space, so we use the pid give that Python multiprocessing spawns a
        # process for each worker.
        pid = os.getpid()
        with VsimemFile(f"/vsimem/src_{pid}.tif") as src_data_path:
            # I don't think this is very safe, but for now this is a place to start
            self._src._set_window(expanded_src_read_area)
            self._src.to_geotiff(src_data_path)
            # Yeah, this is broken, as the set window forms part of the CSE hash, and
            # because this object's CSE hash depends on the _src CSE hash, we really do
            # break things with this with the _set_window, hence this reset_window.
            self._src.reset_window()

            with VsimemFile(f"/vsimem/warped_{pid}.tif") as warped_data_path:
                gdal.Warp(
                    warped_data_path,
                    src_data_path,
                    options=gdal.WarpOptions(
                        dstSRS=projection._gdal_projection,
                        outputType=self.datatype.to_gdal(),
                        xRes=projection.xstep,
                        yRes=projection.ystep,
                        resampleAlg=self._method,
                        targetAlignedPixels=True,
                    )
                )

                with read_raster(warped_data_path) as warped:
                    warped.set_window_for_intersection(read_area)
                    return warped._read_array(0, 0, xsize, ysize)
