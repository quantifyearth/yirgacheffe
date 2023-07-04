import math
from typing import Any, Optional

import numpy
from osgeo import gdal

from .. import WSG_84_PROJECTION
from ..window import Area, PixelScale, Window
from ..rounding import round_up_pixels
from .base import YirgacheffeLayer


class RasterLayer(YirgacheffeLayer):
    """Layer provides a wrapper around a gdal dataset/band that also records offset state so that
    we can work with maps over different geographic regions but work withing a particular frame
    of reference."""

    @staticmethod
    def empty_raster_layer(
        area: Area,
        scale: PixelScale,
        datatype: int,
        filename: Optional[str]=None,
        projection: str=WSG_84_PROJECTION,
        name: Optional[str]=None,
        compress: bool=True
    ):
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)

        # We treat the provided area as aspirational, and we need to align it to pixel boundaries
        pixel_friendly_area = Area(
            left=math.floor(area.left / abs_xstep) * abs_xstep,
            right=math.ceil(area.right / abs_xstep) * abs_xstep,
            top=math.ceil(area.top / abs_ystep) * abs_ystep,
            bottom=math.floor(area.bottom / abs_ystep) * abs_ystep,
        )

        if filename:
            driver = gdal.GetDriverByName('GTiff')
        else:
            driver = gdal.GetDriverByName('mem')
            filename = 'mem'
        dataset = driver.Create(
            filename,
            round_up_pixels((pixel_friendly_area.right - pixel_friendly_area.left) / abs_xstep, abs_xstep),
            round_up_pixels((pixel_friendly_area.top - pixel_friendly_area.bottom) / abs_ystep, abs_ystep),
            1,
            datatype,
            [] if not compress else ['COMPRESS=LZW'],
        )
        dataset.SetGeoTransform([
            pixel_friendly_area.left, scale.xstep, 0.0, pixel_friendly_area.top, 0.0, scale.ystep
        ])
        dataset.SetProjection(projection)
        return RasterLayer(dataset, name=name)

    @staticmethod
    def empty_raster_layer_like(
        layer: YirgacheffeLayer,
        filename: Optional[str]=None,
        area: Optional[Area]=None,
        datatype: Optional[int]=None,
        compress: bool=True
    ):
        width = layer.window.xsize
        height = layer.window.ysize
        geo_transform = layer.geo_transform
        if area is not None:
            scale = layer.pixel_scale
            if scale is None:
                raise ValueError("Can not work out area without explicit pixel scale")
            abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
            width = round_up_pixels((area.right - area.left) / abs_xstep, abs_xstep)
            height = round_up_pixels((area.top - area.bottom) / abs_ystep, abs_ystep)
            geo_transform = (
                area.left, scale.xstep, 0.0, area.top, 0.0, scale.ystep
            )

        if filename:
            driver = gdal.GetDriverByName('GTiff')
        else:
            driver = gdal.GetDriverByName('mem')
            filename = 'mem'
        dataset = driver.Create(
            filename,
            width,
            height,
            1,
            datatype if datatype is not None else layer.datatype,
            [] if not compress else ['COMPRESS=LZW'],
        )
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(layer.projection)
        return RasterLayer(dataset)

    @classmethod
    def layer_from_file(cls, filename: str):
        dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(filename)
        return cls(dataset, filename)

    def __init__(self, dataset: gdal.Dataset, name: Optional[str] = None):
        if not dataset:
            raise ValueError("None is not a valid dataset")

        transform = dataset.GetGeoTransform()
        scale = PixelScale(transform[1], transform[5])
        area = Area(
            left=transform[0],
            top=transform[3],
            right=transform[0] + (dataset.RasterXSize * scale.xstep),
            bottom=transform[3] + (dataset.RasterYSize * scale.ystep),
        )

        super().__init__(
            area,
            scale,
            dataset.GetProjection()
        )

        self._dataset = dataset
        self._raster_xsize = dataset.RasterXSize
        self._raster_ysize = dataset.RasterYSize
        self.name = name
        # default window to full layer
        self.window = Window(
            xoff=0,
            yoff=0,
            xsize=dataset.RasterXSize,
            ysize=dataset.RasterYSize,
        )

    @property
    def datatype(self) -> int:
        return self._dataset.GetRasterBand(1).DataType

    def read_array(self, xoffset, yoffset, xsize, ysize) -> Any:
        # if we're dealing with an intersection, we can just read the data directly,
        # otherwise we need to read the data into another array with suitable padding
        target_window = Window(
            self.window.xoff + xoffset,
            self.window.yoff + yoffset,
            xsize,
            ysize
        )
        source_window = Window(
            xoff=0,
            yoff=0,
            xsize=self._raster_xsize,
            ysize=self._raster_ysize,
        )
        try:
            intersection = Window.find_intersection([source_window, target_window])
        except ValueError:
            return numpy.zeros((ysize, xsize))

        if target_window == intersection:
            # The target window is a subset of or equal to the source, so we can just ask for the data
            data = self._dataset.GetRasterBand(1).ReadAsArray(*intersection.as_array_args)
            return data
        else:
            # We should read the intersection from the array, and the rest should be zeros
            subset = self._dataset.GetRasterBand(1).ReadAsArray(*intersection.as_array_args)
            data = numpy.pad(
                subset,
                (
                    (
                        (intersection.yoff - self.window.yoff) - yoffset,
                        (ysize - ((intersection.yoff - self.window.yoff) + intersection.ysize)) + yoffset,
                    ),
                    (
                        (intersection.xoff - self.window.xoff) - xoffset,
                        xsize - ((intersection.xoff - self.window.xoff) + intersection.xsize) + xoffset,
                    )
                ),
                'constant'
            )
            return data
