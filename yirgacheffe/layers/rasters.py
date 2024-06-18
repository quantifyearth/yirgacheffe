import math
import os
from typing import Any, Optional, TypeVar, Union

import numpy
from osgeo import gdal

from .. import WGS_84_PROJECTION
from ..window import Area, PixelScale, Window
from ..rounding import round_up_pixels
from .base import YirgacheffeLayer

# Still to early to require Python 3.11 :/
RasterLayerT = TypeVar("RasterLayerT", bound="RasterLayer")

class InvalidRasterBand(Exception):
    def __init__ (self, band):
        self.band = band

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
        projection: str=WGS_84_PROJECTION,
        name: Optional[str]=None,
        compress: bool=True,
        nodata: Optional[Union[float,int]]=None,
        nbits: Optional[int]=None
    ) -> RasterLayerT:
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)

        # We treat the provided area as aspirational, and we need to align it to pixel boundaries
        pixel_friendly_area = Area(
            left=math.floor(area.left / abs_xstep) * abs_xstep,
            right=math.ceil(area.right / abs_xstep) * abs_xstep,
            top=math.ceil(area.top / abs_ystep) * abs_ystep,
            bottom=math.floor(area.bottom / abs_ystep) * abs_ystep,
        )

        options = []
        if nbits is not None:
            options.append(f"NBITS={nbits}")

        if filename:
            driver = gdal.GetDriverByName('GTiff')
            options.append('BIGTIFF=YES')
            if compress:
                options.append('COMPRESS=LZW')
        else:
            driver = gdal.GetDriverByName('mem')
            filename = 'mem'
            compress = False
        dataset = driver.Create(
            filename,
            round_up_pixels((pixel_friendly_area.right - pixel_friendly_area.left) / abs_xstep, abs_xstep),
            round_up_pixels((pixel_friendly_area.top - pixel_friendly_area.bottom) / abs_ystep, abs_ystep),
            1,
            datatype,
            options
        )
        dataset.SetGeoTransform([
            pixel_friendly_area.left, scale.xstep, 0.0, pixel_friendly_area.top, 0.0, scale.ystep
        ])
        dataset.SetProjection(projection)
        if nodata is not None:
            dataset.GetRasterBand(1).SetNoDataValue(nodata)
        return RasterLayer(dataset, name=name)

    @staticmethod
    def empty_raster_layer_like(
        layer: YirgacheffeLayer,
        filename: Optional[str]=None,
        area: Optional[Area]=None,
        datatype: Optional[int]=None,
        compress: bool=True,
        nodata: Optional[Union[float,int]]=None,
        nbits: Optional[int]=None
    ) -> RasterLayerT:
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

        options = []
        if nbits is not None:
            options.append(f"NBITS={nbits}")

        if filename:
            driver = gdal.GetDriverByName('GTiff')
            options.append('BIGTIFF=YES')
            if compress:
                options.append('COMPRESS=LZW')
        else:
            driver = gdal.GetDriverByName('mem')
            filename = 'mem'
            compress = False
        dataset = driver.Create(
            filename,
            width,
            height,
            1,
            datatype if datatype is not None else layer.datatype,
            options,
        )
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(layer.projection)
        if nodata is not None:
            dataset.GetRasterBand(1).SetNoDataValue(nodata)

        return RasterLayer(dataset)

    @classmethod
    def scaled_raster_from_raster(
        cls,
        source: RasterLayerT,
        new_pixel_scale: PixelScale,
        filename: Optional[str]=None,
        compress: bool=True,
        algorithm: int=gdal.GRA_NearestNeighbour,
    ) -> RasterLayerT:
        source_dataset = source._dataset
        old_pixel_scale = source.pixel_scale
        assert old_pixel_scale

        x_scale = old_pixel_scale.xstep / new_pixel_scale.xstep
        y_scale = old_pixel_scale.ystep / new_pixel_scale.ystep
        new_width = round_up_pixels(source_dataset.RasterXSize * x_scale,
            abs(new_pixel_scale.xstep))
        new_height = round_up_pixels(source_dataset.RasterYSize * y_scale,
            abs(new_pixel_scale.ystep))

        # in yirgacheffe we like to have things aligned to the pixel_scale, so work
        # out new top left corner
        new_left = math.floor((source.area.left / new_pixel_scale.xstep)) * new_pixel_scale.xstep
        new_top = math.ceil((source.area.top / new_pixel_scale.ystep)) * new_pixel_scale.ystep

        # now build a target dataset
        options = []
        if filename:
            driver = gdal.GetDriverByName('GTiff')
            options.append('BIGTIFF=YES')
            if compress:
                options.append('COMPRESS=LZW')
        else:
            driver = gdal.GetDriverByName('mem')
            filename = 'mem'
            compress = False
        dataset = driver.Create(
            filename,
            new_width,
            new_height,
            1,
            source.datatype,
            options
        )
        dataset.SetGeoTransform((
            new_left, new_pixel_scale.xstep, 0.0,
            new_top, 0.0, new_pixel_scale.ystep
        ))
        dataset.SetProjection(source_dataset.GetProjection())

        # now use gdal to do the reprojection
        gdal.ReprojectImage(source_dataset, dataset, eResampleAlg=algorithm)

        return RasterLayer(dataset)

    @classmethod
    def layer_from_file(cls, filename: str, band: int = 1) -> RasterLayerT:
        try:
            dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        except RuntimeError as exc:
            # With exceptions on GDAL now returns the wrong (IMHO) exception
            raise FileNotFoundError(filename) from exc
        try:
            _ = dataset.GetRasterBand(band)
        except RuntimeError as exc:
            raise InvalidRasterBand(band) from exc
        return cls(dataset, filename, band)

    def __init__(self, dataset: gdal.Dataset, name: Optional[str] = None, band: int = 1):
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
            dataset.GetProjection(),
            name=name
        )

        # The constructor works out the window from the area
        # so sanity check that the calculated window matches the
        # dataset's dimensions
        assert self.window == Window(0, 0, dataset.RasterXSize, dataset.RasterYSize)

        self._dataset = dataset
        self._dataset_path = dataset.GetDescription()
        self._band = band
        self._raster_xsize = dataset.RasterXSize
        self._raster_ysize = dataset.RasterYSize

    def __getstate__(self) -> object:
        # Only support pickling on file backed layers (ideally read only ones...)
        if not os.path.isfile(self._dataset_path):
            raise ValueError("Can not pickle layer that is not file backed.")
        odict = self.__dict__.copy()
        self._park()
        del odict['_dataset']
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._unpark()

    def _park(self):
        try:
            self._dataset.Close()
        except AttributeError:
            pass
        self._dataset = None

    def _unpark(self):
        if getattr(self, "_dataset", None) is None:
            try:
                self._dataset = gdal.Open(self._dataset_path)
            except RuntimeError as exc:
                raise FileNotFoundError(f"Failed to open pickled raster {self._dataset_path}") from exc

    @property
    def datatype(self) -> int:
        if self._dataset is None:
            self._unpark()
        return self._dataset.GetRasterBand(1).DataType

    def read_array(self, xoffset, yoffset, xsize, ysize) -> Any:
        if self._dataset is None:
            self._unpark()
        if (xsize <= 0) or (ysize <= 0):
            raise ValueError("Request dimensions must be positive and non-zero")

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
            data = self._dataset.GetRasterBand(self._band).ReadAsArray(*intersection.as_array_args)
            return data
        else:
            # We should read the intersection from the array, and the rest should be zeros
            subset = self._dataset.GetRasterBand(self._band).ReadAsArray(*intersection.as_array_args)
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
