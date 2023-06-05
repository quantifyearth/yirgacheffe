import math
from collections import namedtuple
from math import ceil, floor
from typing import Any, List, Optional, Tuple

import numpy
from osgeo import gdal, ogr

from . import WSG_84_PROJECTION
from .window import Area, Window
from .operators import LayerMathMixin
from .rounding import almost_equal, round_up_pixels, round_down_pixels

PixelScale = namedtuple('PixelScale', ['xstep', 'ystep'])


class YirgacheffeLayer(LayerMathMixin):

    @staticmethod
    def find_intersection(layers: List) -> Area:
        if not layers:
            raise ValueError("Expected list of layers")

        # This only makes sense (currently) if all layers
        # have the same pixel pitch
        scale = layers[0].pixel_scale
        for layer in layers[1:]:
            if not layer.check_pixel_scale(scale):
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
        # have the same pixel pitch
        scale = layers[0].pixel_scale
        for layer in layers[1:]:
            if not layer.check_pixel_scale(scale):
                raise ValueError("Not all layers are at the same pixel scale")

        return Area(
            left=min(x.area.left for x in layers),
            top=max(x.area.top for x in layers),
            right=max(x.area.right for x in layers),
            bottom=min(x.area.bottom for x in layers)
        )

    @property
    def geo_transform(self) -> Tuple[float, float, float, float, float, float]:
        if self._intersection:
            return (
                self._intersection.left, self._transform[1],
                0.0, self._intersection.top, 0.0, self._transform[5]
            )
        return self._transform

    @property
    def pixel_scale(self) -> Optional[PixelScale]:
        return PixelScale(self._transform[1], self._transform[5])

    def check_pixel_scale(self, scale: PixelScale) -> bool:
        our_scale = self.pixel_scale
        assert our_scale is not None
        return almost_equal(our_scale.xstep, scale.xstep) and \
            almost_equal(our_scale.ystep, scale.ystep)

    def set_window_for_intersection(self, intersection: Area) -> None:
        new_window = Window(
            xoff=round_down_pixels((intersection.left - self.area.left) / self._transform[1],
                self._transform[1]),
            yoff=round_down_pixels((self.area.top - intersection.top) / (self._transform[5] * -1.0),
                self._transform[5] * -1.0),
            xsize=round_up_pixels(
                (intersection.right - intersection.left) / self._transform[1],
                self._transform[1]
            ),
            ysize=round_up_pixels(
                (intersection.top - intersection.bottom) / (self._transform[5] * -1.0),
                (self._transform[5] * -1.0)
            ),
        )
        if (new_window.xoff < 0) or (new_window.yoff < 0):
            raise ValueError('Window has negative offset')
        if self._dataset:
            if ((new_window.xoff + new_window.xsize) > self._raster_xsize) or \
                ((new_window.yoff + new_window.ysize) > self._raster_ysize):
                raise ValueError(f'Window is bigger than dataset: raster is {self._raster_xsize}x{self._raster_ysize}'\
                    f', new window is {new_window.xsize - new_window.xoff}x{new_window.ysize - new_window.yoff}')
        self.window = new_window
        self._intersection = intersection

    def set_window_for_union(self, intersection: Area) -> None:
        new_window = Window(
            xoff=round_down_pixels((intersection.left - self.area.left) / self._transform[1],
                self._transform[1]),
            yoff=round_down_pixels((self.area.top - intersection.top) / (self._transform[5] * -1.0),
                self._transform[5] * -1.0),
            xsize=round_up_pixels(
                (intersection.right - intersection.left) / self._transform[1],
                self._transform[1]
            ),
            ysize=round_up_pixels(
                (intersection.top - intersection.bottom) / (self._transform[5] * -1.0),
                (self._transform[5] * -1.0)
            ),
        )
        if (new_window.xoff > 0) or (new_window.yoff > 0):
            raise ValueError('Window has positive offset')
        if self._dataset:
            if ((new_window.xsize - new_window.xoff) < self._raster_xsize) or \
                ((new_window.ysize - new_window.yoff) < self._raster_ysize):
                raise ValueError(f'Window is smaller than dataset: raster is {self._raster_xsize}x{self._raster_ysize}'\
                    f', new window is {new_window.xsize - new_window.xoff}x{new_window.ysize - new_window.yoff}')
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


class Layer(YirgacheffeLayer):
    """Layer provides a wrapper around a gdal dataset/band that also records offset state so that
    we can work with maps over different geographic regions but work withing a particular frame
    of reference."""

    @staticmethod
    def empty_raster_layer(
        area: Area,
        scale: PixelScale,
        data_type: int,
        filename: Optional[str]=None,
        projection: str=WSG_84_PROJECTION,
        name: Optional[str]=None,
        compress: bool=False
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
            data_type,
            [] if not compress else ['COMPRESS=LZW'],
        )
        dataset.SetGeoTransform([
            pixel_friendly_area.left, scale.xstep, 0.0, pixel_friendly_area.top, 0.0, scale.ystep
        ])
        dataset.SetProjection(projection)
        return Layer(dataset, name=name)

    @staticmethod
    def empty_raster_layer_like(layer, filename: Optional[str]=None, datatype: Optional[int]=None):
        if filename:
            driver = gdal.GetDriverByName('GTiff')
        else:
            driver = gdal.GetDriverByName('mem')
            filename = 'mem'
        dataset = driver.Create(
            filename,
            layer.window.xsize,
            layer.window.ysize,
            1,
            datatype if datatype is not None else layer.datatype,
            [] if filename == 'mem' else ['COMPRESS=LZW'],
        )
        dataset.SetGeoTransform(layer.geo_transform)
        dataset.SetProjection(layer.projection)
        return Layer(dataset)

    @classmethod
    def layer_from_file(cls, filename: str):
        dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(filename)
        return cls(dataset, filename)

    def __init__(self, dataset: gdal.Dataset, name: Optional[str] = None):
        if not dataset:
            raise ValueError("None is not a valid dataset")
        self._dataset = dataset
        self._transform = dataset.GetGeoTransform()
        self._raster_xsize = dataset.RasterXSize
        self._raster_ysize = dataset.RasterYSize
        self._intersection: Optional[Area] = None
        self.name = name

        # Global position of the layer
        self.area = Area(
            left=self._transform[0],
            top=self._transform[3],
            right=self._transform[0] + (self._raster_xsize * self._transform[1]),
            bottom=self._transform[3] + (self._raster_ysize * self._transform[5]),
        )

        # default window to full layer
        self.window = Window(
            xoff=0,
            yoff=0,
            xsize=dataset.RasterXSize,
            ysize=dataset.RasterYSize,
        )

    @property
    def projection(self) -> str:
        return self._dataset.GetProjection()

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


class RasterLayer(Layer):
    """A place holder for now, at some point I want to replace Layer with RasterLayer."""


class RasteredVectorLayer(Layer):
    """This layer takes a vector file and rasterises it for the given filter. Rasterization
    up front like this is very expensive, so not recommended. Instead you should use
    VectorLayer."""

    @classmethod
    def layer_from_file(cls, filename: str, where_filter: Optional[str], scale: PixelScale, projection: str): # type: ignore
        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)
        vector_layer = RasteredVectorLayer(layer, scale, projection)

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        return vector_layer

    def __init__(self, layer: ogr.Layer, scale: PixelScale, projection: str):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer

        # work out region for mask
        envelopes = []
        layer.ResetReading()
        feature = layer.GetNextFeature()
        while feature:
            envelopes.append(feature.GetGeometryRef().GetEnvelope())
            feature = layer.GetNextFeature()
        if len(envelopes) == 0:
            raise ValueError('No geometry found for')

        # Get the area, but scale it to the pixel resolution that we're using. Note that
        # the pixel scale GDAL uses can have -ve values, but those will mess up the
        # ceil/floor math, so we use absolute versions when trying to round.
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
        area = Area(
            left=floor(min(x[0] for x in envelopes) / abs_xstep) * abs_xstep,
            top=ceil(max(x[3] for x in envelopes) / abs_ystep) * abs_ystep,
            right=ceil(max(x[1] for x in envelopes) / abs_xstep) * abs_xstep,
            bottom=floor(min(x[2] for x in envelopes) / abs_ystep) * abs_ystep,
        )

        # create new dataset for just that area
        dataset = gdal.GetDriverByName('mem').Create(
            'mem',
            round((area.right - area.left) / abs_xstep),
            round((area.top - area.bottom) / abs_ystep),
            1,
            gdal.GDT_Byte,
            []
        )
        if not dataset:
            raise MemoryError('Failed to create memory mask')

        dataset.SetProjection(projection)
        dataset.SetGeoTransform([area.left, scale.xstep, 0.0, area.top, 0.0, scale.ystep])
        gdal.RasterizeLayer(dataset, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        super().__init__(dataset)


class VectorRangeLayer(RasteredVectorLayer):
    """Deprecated older name for VectorLayer"""

    def __init__(self, range_vectors: str, where_filter: str, scale: PixelScale, projection: str):
        vectors = ogr.Open(range_vectors)
        if vectors is None:
            raise FileNotFoundError(range_vectors)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)
        super().__init__(layer, scale, projection)


class VectorLayer(Layer):
    """This layer takes a vector file and rasterises it for the given filter. Rasterization occurs only
    when the data is fetched, so there is no explosive memeory cost, but fetching small units (e.g., one
    line at a time) can be quite slow, so recommended that you fetch reasonable chunks each time (or
    modify this class so that it chunks things internally)."""

    @classmethod
    def layer_from_file(cls, filename: str, where_filter: Optional[str], scale: PixelScale, projection: str): # type: ignore
        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)
        vector_layer = VectorLayer(layer, scale, projection)

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        return vector_layer


    def __init__(self, layer: ogr.Layer, scale: PixelScale, projection: str):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer

        # work out region for mask
        envelopes = []
        layer.ResetReading()
        feature = layer.GetNextFeature()
        while feature:
            envelopes.append(feature.GetGeometryRef().GetEnvelope())
            feature = layer.GetNextFeature()
        if len(envelopes) == 0:
            raise ValueError('No geometry found')

        # Get the area, but scale it to the pixel resolution that we're using. Note that
        # the pixel scale GDAL uses can have -ve values, but those will mess up the
        # ceil/floor math, so we use absolute versions when trying to round.
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)
        self.area = Area(
            left=floor(min(x[0] for x in envelopes) / abs_xstep) * abs_xstep,
            top=ceil(max(x[3] for x in envelopes) / abs_ystep) * abs_ystep,
            right=ceil(max(x[1] for x in envelopes) / abs_xstep) * abs_xstep,
            bottom=floor(min(x[2] for x in envelopes) / abs_ystep) * abs_ystep,
        )
        self._transform = [self.area.left, scale.xstep, 0.0, self.area.top, 0.0, scale.ystep]
        self._projection = projection
        self._dataset = None
        self._intersection = self.area
        self.window = Window(
            xoff=0,
            yoff=0,
            xsize=ceil((self.area.right - self.area.left) / scale.xstep),
            ysize=ceil((self.area.bottom - self.area.top) / scale.ystep),
        )

    @property
    def projection(self) -> str:
        return self._projection

    @property
    def datatype(self) -> int:
        return gdal.GDT_Byte

    def read_array(self, xoffset, yoffset, xsize, ysize):

        # I did try recycling this object to save allocation/dealloction, but in practice it
        # seemed to only make things slower (particularly as you need to zero the memory each time yourself)
        dataset = gdal.GetDriverByName('mem').Create(
            'mem',
            xsize,
            ysize,
            1,
            gdal.GDT_Byte,
            []
        )
        if not dataset:
            raise MemoryError('Failed to create memory mask')

        dataset.SetProjection(self._projection)
        dataset.SetGeoTransform([
            self._intersection.left + (xoffset * self._transform[1]), self._transform[1], 0.0,
            self._intersection.top + (yoffset * self._transform[5]), 0.0, self._transform[5]
        ])
        gdal.RasterizeLayer(dataset, [1], self.layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])

        res = dataset.ReadAsArray(0, 0, xsize, ysize)
        return res


class DynamicVectorRangeLayer(VectorLayer):
    """Deprecated older name DynamicVectorLayer"""

    def __init__(self, range_vectors: str, where_filter: str, scale: PixelScale, projection: str):
        vectors = ogr.Open(range_vectors)
        if vectors is None:
            raise FileNotFoundError(range_vectors)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)
        super().__init__(layer, scale, projection)


class UniformAreaLayer(Layer):
    """If you have a pixel area map where all the row entries are identical, then you
    can speed up the AoH calculations by simplifying that to a 1 pixel wide map and then
    synthesizing the rest of the data at calc time, as decompressing the large compressed
    TIFF files is quite slow. This class is used to load such a dataset.

    If you have a file that is large that you'd like to shrink you can call the static methid
    generate_narrow_area_projection which will shrink the file and correct the geo info.
    """

    @staticmethod
    def generate_narrow_area_projection(source_filename: str, target_filename: str) -> None:
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

    def __init__(self, dataset, name: Optional[str] = None):
        if dataset.RasterXSize > 1:
            raise ValueError("Expected a shrunk dataset")
        self.databand = dataset.GetRasterBand(1).ReadAsArray(0, 0, 1, dataset.RasterYSize)

        super().__init__(dataset, name)

        transform = dataset.GetGeoTransform()
        self.window = Window(
            xoff=0,
            yoff=0,
            xsize=360 / transform[1],
            ysize=dataset.RasterYSize,
        )
        self._raster_xsize = self.window.xsize
        self.area = Area(-180, self.area.top, 180, self.area.bottom)

    def read_array(self, xoffset, yoffset, _xsize, ysize) -> Any:
        offset = self.window.yoff + yoffset
        return self.databand[offset:offset + ysize]


class ConstantLayer(Layer):
    """This is a layer that will return the identity value - can be used when an input layer is
    missing (e.g., area) without having the calculation full of branches."""
    def __init__(self, value): # pylint: disable=W0231
        self.value = value
        self.area = Area(
            left = -180.0,
            top = 90.0,
            right = 180.0,
            bottom = -90.0
        )

    @property
    def pixel_scale(self) -> Optional[PixelScale]:
        return None

    @property
    def datatype(self) -> int:
        return gdal.GDT_CFloat64

    def check_pixel_scale(self, _scale: PixelScale) -> bool:
        return True

    def set_window_for_intersection(self, _intersection: Area) -> None:
        pass

    def read_array(self, _x: int, _y: int, _xsize: int, _ysize: int) -> Any:
        return self.value
