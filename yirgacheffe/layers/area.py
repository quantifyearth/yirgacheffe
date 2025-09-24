from __future__ import annotations

from math import ceil, floor
from pathlib import Path
from typing import Any

import numpy
from osgeo import gdal

from ..window import Area, Window
from .rasters import RasterLayer

class UniformAreaLayer(RasterLayer):
    """If you have a pixel area map where all the row entries are identical, then you
    can speed up the AoH calculations by simplifying that to a 1 pixel wide map and then
    synthesizing the rest of the data at calc time, as decompressing the large compressed
    TIFF files is quite slow. This class is used to load such a dataset.

    If you have a file that is large that you'd like to shrink you can call the static method
    generate_narrow_area_projection which will shrink the file and correct the geo info.
    """

    @staticmethod
    def generate_narrow_area_projection(source_filename: Path | str, target_filename: Path | str) -> None:
        source = gdal.Open(source_filename, gdal.GA_ReadOnly)
        if source is None:
            raise FileNotFoundError(source_filename)
        if not UniformAreaLayer.is_uniform_area_projection(source):
            raise ValueError("Data in area pixel map is not uniform across rows")
        source_band = source.GetRasterBand(1)
        target = gdal.GetDriverByName('GTiff').Create(
            target_filename,
            1,
            source.RasterYSize,
            1,
            source_band.DataType,
            ['COMPRESS=LZW']
        )
        target.SetProjection(source.GetProjection())
        target.SetGeoTransform(source.GetGeoTransform())
        # Although the output is 1 pixel wide, the input can be very wide, so we do this in stages
        # otherwise gdal eats all the memory
        step = 1000
        target_band = target.GetRasterBand(1)
        for yoffset in range(0, source.RasterYSize, step):
            this_step = step
            if (yoffset + this_step) > source.RasterYSize:
                this_step = source.RasterYSize - yoffset
            data = source_band.ReadAsArray(0, yoffset, 1, this_step)
            target_band.WriteArray(data, 0, yoffset)

    @staticmethod
    def is_uniform_area_projection(dataset) -> bool:
        "Check that the dataset conforms to the assumption that all rows contain the same value. Likely to be slow."
        band = dataset.GetRasterBand(1)
        for yoffset in range(dataset.RasterYSize):
            row = band.ReadAsArray(0, yoffset, dataset.RasterXSize, 1)
            if not numpy.all(numpy.isclose(row, row[0])):
                return False
        return True

    def __init__(self, dataset, name: str | None = None, band: int = 1, ignore_nodata: bool = False):
        if dataset.RasterXSize > 1:
            raise ValueError("Expected a shrunk dataset")
        self.databand = dataset.GetRasterBand(1).ReadAsArray(0, 0, 1, dataset.RasterYSize)

        super().__init__(dataset, name, band, ignore_nodata)

        transform = dataset.GetGeoTransform()

        projection = self.map_projection
        assert projection is not None # from raster we should always have one

        self._underlying_area = Area(
            floor(-180 / projection.xstep) * projection.xstep,
            self.area.top,
            ceil(180 / projection.xstep) * projection.xstep,
            self.area.bottom
        )
        self._active_area = self._underlying_area

        self._window = Window(
            xoff=0,
            yoff=0,
            xsize=int((self.area.right - self.area.left) / transform[1]),
            ysize=dataset.RasterYSize,
        )
        self._raster_xsize = self.window.xsize

    def _read_array_with_window(
        self,
        xoffset: int,
        yoffset: int,
        xsize: int,
        ysize: int,
        window: Window,
    ) -> Any:
        if ysize <= 0:
            raise ValueError("Request dimensions must be positive and non-zero")
        offset = window.yoff + yoffset
        return self.databand[offset:offset + ysize]
