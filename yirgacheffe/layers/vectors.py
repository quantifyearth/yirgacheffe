from __future__ import annotations
import os
from math import ceil, floor
from typing import Any, Optional, Tuple, Union
from typing_extensions import NotRequired

from osgeo import gdal, ogr

from ..operators import DataType
from ..window import Area, PixelScale
from .base import YirgacheffeLayer
from .rasters import RasterLayer
from .._backends import backend

def _validate_burn_value(burn_value: Any, layer: ogr.Layer) -> DataType: # pylint: disable=R0911
    if isinstance(burn_value, str):
        # burn value is field name, so validate it
        index = layer.FindFieldIndex(burn_value, True)
        if index < 0:
            raise ValueError("Burn value not found as field")
        # if the user hasn't specified, pick datatype from
        # fiend definition.
        definition = layer.GetLayerDefn()
        field = definition.GetFieldDefn(index)
        typename = field.GetTypeName()
        if typename == "Integer":
            return DataType.Int64
        elif typename == "Real":
            return DataType.Float64
        else:
            raise ValueError(f"Can't set datatype {typename} for burn value {burn_value}")
    elif isinstance(burn_value, int):
        if 0 <= burn_value <= 255:
            return DataType.Byte
        else:
            unsigned = burn_value > 0
            if unsigned:
                if burn_value < (pow(2, 16)):
                    return DataType.UInt16
                elif burn_value < (pow(2, 32)):
                    return DataType.UInt32
                else:
                    return DataType.UInt64
            else:
                if abs(burn_value) < (pow(2, 15)):
                    return DataType.Int16
                elif abs(burn_value) < (pow(2, 31)):
                    return DataType.Int32
                else:
                    return DataType.Int64
    elif isinstance(burn_value, float):
        return DataType.Float64
    else:
        raise ValueError(f"data type of burn value {burn_value} not supported")


class RasteredVectorLayer(RasterLayer):
    """This layer takes a vector file and rasterises it for the given filter. Rasterization
    up front like this is very expensive, so not recommended. Instead you should use
    VectorLayer."""

    @classmethod
    def layer_from_file( # type: ignore[override] # pylint: disable=W0221
        cls,
        filename: str,
        where_filter: Optional[str],
        scale: PixelScale,
        projection: str,
        datatype: Optional[Union[int, DataType]] = None,
        burn_value: Union[int,float,str] = 1,
    ) -> RasteredVectorLayer:
        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)

        estimated_datatype = _validate_burn_value(burn_value, layer)
        if datatype is None:
            datatype_arg: DataType = estimated_datatype
        elif isinstance(datatype, int):
            datatype_arg = DataType.of_gdal(datatype)
        else:
            datatype_arg = datatype

        vector_layer = RasteredVectorLayer(
            layer,
            scale,
            projection,
            datatype=datatype_arg,
            burn_value=burn_value
        )

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        return vector_layer

    def __init__(
        self,
        layer: ogr.Layer,
        scale: PixelScale,
        projection: str,
        datatype: Union[int, DataType] = DataType.Byte,
        burn_value: Union[int,float,str] = 1,
    ):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer

        self._original: Optional[Any] = None

        if isinstance(datatype, int):
            datatype_arg = DataType.of_gdal(datatype)
        else:
            datatype_arg = datatype

        # work out region for mask
        envelopes = []
        layer.ResetReading()
        feature = layer.GetNextFeature()
        while feature:
            geometry = feature.GetGeometryRef()
            if geometry:
                envelopes.append(geometry.GetEnvelope())
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
            datatype_arg.to_gdal(),
            []
        )
        if not dataset:
            raise MemoryError('Failed to create memory mask')

        dataset.SetProjection(projection)
        dataset.SetGeoTransform([area.left, scale.xstep, 0.0, area.top, 0.0, scale.ystep])
        if isinstance(burn_value, (int, float)):
            gdal.RasterizeLayer(dataset, [1], self.layer, burn_values=[burn_value], options=["ALL_TOUCHED=TRUE"])
        elif isinstance(burn_value, str):
            gdal.RasterizeLayer(dataset, [1], self.layer, options=[f"ATTRIBUTE={burn_value}", "ALL_TOUCHED=TRUE"])
        else:
            raise ValueError("Burn value for layer should be number or field name")
        super().__init__(dataset)


class VectorLayer(YirgacheffeLayer):
    """This layer takes a vector file and rasterises it for the given filter. Rasterization occurs only
    when the data is fetched, so there is no explosive memeory cost, but fetching small units (e.g., one
    line at a time) can be quite slow, so recommended that you fetch reasonable chunks each time (or
    modify this class so that it chunks things internally)."""

    @classmethod
    def layer_from_file_like(
        cls,
        filename: str,
        other_layer: YirgacheffeLayer,
        where_filter: Optional[str]=None,
        datatype: Optional[Union[int, DataType]] = None,
        burn_value: Union[int,float,str] = 1,
    ) -> VectorLayer:
        if other_layer is None:
            raise ValueError("like layer can not be None")
        if other_layer.pixel_scale is None:
            raise ValueError("Reference layer must have pixel scale")

        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)

        if datatype is not None:
            if isinstance(datatype, int):
                datatype = DataType.of_gdal(datatype)

        vector_layer = VectorLayer(
            layer,
            other_layer.pixel_scale,
            other_layer.projection,
            name=filename,
            datatype=datatype if datatype is not None else other_layer.datatype,
            burn_value=burn_value,
            anchor=(other_layer.area.left, other_layer.area.top),
        )

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        vector_layer._dataset_path = filename
        vector_layer._filter = where_filter
        return vector_layer

    @classmethod
    def layer_from_file(
        cls,
        filename: str,
        where_filter: Optional[str],
        scale: PixelScale,
        projection: str,
        datatype: Optional[Union[int, DataType]] = None,
        burn_value: Union[int,float,str] = 1,
        anchor: Tuple[float,float] = (0.0, 0.0)
    ) -> VectorLayer:
        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)

        estimated_datatype = _validate_burn_value(burn_value, layer)
        if datatype is None:
            datatype = estimated_datatype

        if isinstance(datatype, int):
            datatype_arg = DataType.of_gdal(datatype)
        else:
            datatype_arg = datatype

        vector_layer = VectorLayer(
            layer,
            scale,
            projection,
            name=filename,
            datatype=datatype_arg,
            burn_value=burn_value,
            anchor=anchor
        )

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        vector_layer._dataset_path = filename
        vector_layer._filter = where_filter
        return vector_layer

    def __init__(
        self,
        layer: ogr.Layer,
        scale: PixelScale,
        projection: str,
        name: Optional[str] = None,
        datatype: Union[int,DataType] = DataType.Byte,
        burn_value: Union[int,float,str] = 1,
        anchor: Tuple[float,float] = (0.0, 0.0)
    ):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer
        self.name = name

        if isinstance(datatype, int):
            self._datatype = DataType.of_gdal(datatype)
        else:
            self._datatype = datatype

        # If the burn value is a number, use it directly, if it's a string
        # then assume it is a column name in the dataset
        self.burn_value = burn_value

        self._original = None
        self._dataset_path: Optional[str] = None
        self._filter: Optional[str] = None

        # work out region for mask
        envelopes = []
        layer.ResetReading()
        feature = layer.GetNextFeature()
        while feature:
            geometry = feature.GetGeometryRef()
            if geometry:
                envelopes.append(geometry.GetEnvelope())
            feature = layer.GetNextFeature()
        if len(envelopes) == 0:
            raise ValueError('No geometry found')

        # Get the area, but scale it to the pixel resolution that we're using. Note that
        # the pixel scale GDAL uses can have -ve values, but those will mess up the
        # ceil/floor math, so we use absolute versions when trying to round.
        abs_xstep, abs_ystep = abs(scale.xstep), abs(scale.ystep)

        # Lacking any other reference, we will make the raster align with
        # (0.0, 0.0), if sometimes we want to align with an existing raster, so if
        # an anchor is specified, ensure we use that as our pixel space alignment
        x_anchor = anchor[0]
        y_anchor = anchor[1]
        left_shift = x_anchor - abs_xstep
        right_shift = x_anchor
        top_shift = y_anchor
        bottom_shift = y_anchor - abs_ystep

        area = Area(
            left=(floor((min(x[0] for x in envelopes) - left_shift) / abs_xstep) * abs_xstep) + left_shift,
            top=(ceil((max(x[3] for x in envelopes) - top_shift) / abs_ystep) * abs_ystep) + top_shift,
            right=(ceil((max(x[1] for x in envelopes) - right_shift) / abs_xstep) * abs_xstep) + right_shift,
            bottom=(floor((min(x[2] for x in envelopes) - bottom_shift) / abs_ystep) * abs_ystep) + bottom_shift,
        )

        super().__init__(area, scale, projection)

    def __getstate__(self) -> object:
        # Only support pickling on file backed layers (ideally read only ones...)
        if self._dataset_path is None or not os.path.isfile(self._dataset_path):
            raise ValueError("Can not pickle layer that is not file backed.")
        odict = self.__dict__.copy()
        del odict['_original']
        del odict['layer']
        return odict

    def __setstate__(self, state):
        vectors = ogr.Open(state['_dataset_path'])
        if vectors is None:
            raise FileNotFoundError(f"Failed to open pickled vectors {state['_dataset_path']}")
        self.__dict__.update(state)
        self._original = vectors
        self.layer = vectors.GetLayer()
        if self._filter is not None:
            self.layer.SetAttributeFilter(self._filter)

    def _park(self):
        self._original = None

    def _unpark(self):
        if getattr(self, "_original", None) is None:
            try:
                self._original = ogr.Open(self._dataset_path)
            except RuntimeError as exc:
                raise FileNotFoundError(f"Failed to open pickled layer {self._dataset_path}") from exc
            self.layer = self._original.GetLayer()
            if self._filter is not None:
                self.layer.SetAttributeFilter(self._filter)

    @property
    def datatype(self) -> DataType:
        return self._datatype

    def read_array_for_area(self, target_area: Area, x: int, y: int, width: int, height: int) -> Any:
        assert self._pixel_scale is not None

        if self._original is None:
            self._unpark()
        if (width <= 0) or (height <= 0):
            raise ValueError("Request dimensions must be positive and non-zero")

        # I did try recycling this object to save allocation/dealloction, but in practice it
        # seemed to only make things slower (particularly as you need to zero the memory each time yourself)
        dataset = gdal.GetDriverByName('mem').Create(
            'mem',
            width,
            height,
            1,
            self.datatype.to_gdal(),
            []
        )
        if not dataset:
            raise MemoryError('Failed to create memory mask')

        dataset.SetProjection(self._projection)
        dataset.SetGeoTransform([
            target_area.left + (x * self._pixel_scale.xstep),
            self._pixel_scale.xstep,
            0.0,
            target_area.top + (y * self._pixel_scale.ystep),
            0.0,
            self._pixel_scale.ystep
        ])
        if isinstance(self.burn_value, (int, float)):
            gdal.RasterizeLayer(dataset, [1], self.layer, burn_values=[self.burn_value], options=["ALL_TOUCHED=TRUE"])
        elif isinstance(self.burn_value, str):
            gdal.RasterizeLayer(dataset, [1], self.layer, options=[f"ATTRIBUTE={self.burn_value}", "ALL_TOUCHED=TRUE"])
        else:
            raise ValueError("Burn value for layer should be number or field name")

        res = backend.promote(dataset.ReadAsArray(0, 0, width, height))
        return res

    def read_array_with_window(self, _x, _y, _width, _height, _window) -> Any:
        assert NotRequired

    def read_array(self, x: int, y: int, width: int, height: int) -> Any:
        return self.read_array_for_area(self._active_area, x, y, width, height)
