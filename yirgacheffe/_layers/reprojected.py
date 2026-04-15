from __future__ import annotations
import uuid
from enum import Enum
from typing import Any

from osgeo import gdal

from .._datatypes import Area, MapProjection, Window
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

class ResamplingMethod(Enum):
    """Resampling methods used in reprojecting rasters.

    This enumeration defines the resampling methods supported by Yirgacheffe.

    Attributes:
        Average: Computes the average of all non-NODATA contributing pixels
        Max: Selects the maximum value among all non-NODATA contributing pixels
        Med: Computes the median of all non-NODATA contributing pixels
        Min: Selects the minimum value among all non-NODATA contributing pixels
        Mode: Selects the most frequently occurring value among contributing pixels
        Nearest: Uses nearest-neighbor sampling (no interpolation)
        RootMeanSquare: Computes the root mean square of all non-NODATA contributing pixels
    """

    Average = "average"
    Max = "max"
    Med = "med"
    Min = "min"
    Mode = "mode"
    Nearest = "nearest"
    RootMeanSquare = "rms"
    Sum = "sum"

    # The commented out ones fail tests due to Yirgacheffe's chunking
    # behaviour and require more work.
    #
    # Bilinear = "bilinear"
    # Cubic = "cubic"
    # CubicSpline = "cubicspline"
    # Lanczos = "lanczos"
    # Q1 = "q1"
    # Q2 = "q2"

class ReprojectedRasterLayer(YirgacheffeLayer):
    """ReprojectedRasterLayer dynamically reprojects a layer.

    This layer wraps another layer and dynamically reprojects it to a different map projection
    and pixel scale. The reprojection is performed lazily - only when data is actually read.

    Thread Safety:
        Individual ReprojectedRasterLayer instances should not be accessed from multiple
        threads simultaneously. For parallel processing, use Python's multiprocessing module
        where each process creates its own layer instances. This is the recommended approach
        for raster processing as it avoids GIL limitations and ensures proper isolation.
    """

    def __init__(
        self,
        src: YirgacheffeLayer,
        target_projection: MapProjection,
        method: ResamplingMethod,
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
            self.projection,
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

        # Resampling will generally require pixels outside the target area so we can say average correctly.
        # Mode resampling requires a larger buffer (3 pixels) to gather sufficient samples for determining
        # the most frequent value, whereas other methods just need a single pixel. Hopefully testing will
        # have sufficient coverage that we catch these as we expand the methods list.
        if self._method in {ResamplingMethod.Mode, ResamplingMethod.Sum}:
            expand_buffer = 3
        else:
            expand_buffer = 1

        # now we want this area in the source projection
        src_projection = self._src.projection
        assert src_projection is not None
        src_read_area = read_area.reproject(src_projection)

        # Note we should typically never go over the edge of the original
        # source material, as different resampling methods react differently to
        # the synthesised zeros that this will admit (e.g., one of min and max
        # would likely get confused).
        expanded_src_read_area = \
            src_read_area.grow(expand_buffer * src_projection.xstep)
        if self._method not in {ResamplingMethod.Sum}:
            expanded_src_read_area = expanded_src_read_area & self._src.area

        # We need some ID that stops us with other parallel workers potentially in the
        # VSIMEM space, so we use a uuid4 that should be close enough to collision free
        # without relying on things like pids.
        fid = uuid.uuid4()
        with VsimemFile(f"/vsimem/src_{fid}.tif") as src_data_path:
            windowed = self._src.as_area(expanded_src_read_area)
            windowed.to_geotiff(src_data_path)

            with VsimemFile(f"/vsimem/warped_{fid}.tif") as warped_data_path:
                gdal.Warp(
                    warped_data_path,
                    src_data_path,
                    options=gdal.WarpOptions(
                        dstSRS=projection._gdal_projection,
                        outputType=self.datatype.to_gdal(),
                        xRes=projection.xstep,
                        yRes=projection.ystep,
                        width=xsize,
                        height=ysize,
                        resampleAlg=self._method.value,
                        targetAlignedPixels=False,
                        outputBounds=(read_area.left, read_area.bottom, read_area.right, read_area.top)
                    )
                )

                with RasterLayer.layer_from_file(warped_data_path) as warped:
                    if (warped._virtual_window.xsize != xsize) or \
                        (warped._virtual_window.ysize != ysize):
                        raise RuntimeError(
                            f"gdal warp violated request constraints: expected {xsize}x{ysize}, "
                            f"got {warped._virtual_window.xsize}x{warped._virtual_window.ysize}"
                        )
                    return warped._read_array(0, 0, xsize, ysize)
