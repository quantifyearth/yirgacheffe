from math import ceil, floor
from typing import Any, Optional, Union

from osgeo import gdal, ogr

from ..window import Area, PixelScale, Window
from .base import YirgacheffeLayer
from .rasters import RasterLayer

def _validate_burn_value(burn_value: Any, layer: ogr.Layer) -> int: # pylint: disable=R0911
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
            return gdal.GDT_Int64
        elif typename == "Real":
            return gdal.GDT_Float64
        else:
            raise ValueError(f"Can't set datatype {typename} for burn value {burn_value}")
    elif isinstance(burn_value, int):
        if 0 <= burn_value <= 255:
            return gdal.GDT_Byte
        else:
            unsigned = burn_value > 0
            if unsigned:
                if burn_value < (pow(2, 16)):
                    return gdal.GDT_UInt16
                elif burn_value < (pow(2, 32)):
                    return gdal.GDT_UInt32
                else:
                    return gdal.GDT_UInt64
            else:
                if abs(burn_value) < (pow(2, 15)):
                    return gdal.GDT_Int16
                elif abs(burn_value) < (pow(2, 31)):
                    return gdal.GDT_Int32
                else:
                    return gdal.GDT_Int64
    elif isinstance(burn_value, float):
        return gdal.GDT_Float64
    else:
        raise ValueError(f"data type of burn value {burn_value} not supported")


class RasteredVectorLayer(RasterLayer):
    """This layer takes a vector file and rasterises it for the given filter. Rasterization
    up front like this is very expensive, so not recommended. Instead you should use
    VectorLayer."""

    @classmethod
    def layer_from_file(
        cls,
        filename: str,
        where_filter: Optional[str],
        scale: PixelScale,
        projection: str,
        datatype: Optional[int] = None,
        burn_value: Union[int,float,str] = 1,
    ): # pylint: disable=W0221
        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)

        estimated_datatype = _validate_burn_value(burn_value, layer)
        if datatype is None:
            datatype = estimated_datatype

        vector_layer = RasteredVectorLayer(
            layer,
            scale,
            projection,
            datatype=datatype,
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
        datatype: int = gdal.GDT_Byte,
        burn_value: Union[int,float,str] = 1,
    ):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer

        self._original = None

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
            datatype,
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
    def layer_from_file(
        cls,
        filename: str,
        where_filter: Optional[str],
        scale: PixelScale,
        projection: str,
        datatype: Optional[int] = None,
        burn_value: Union[int,float,str] = 1,
    ):
        vectors = ogr.Open(filename)
        if vectors is None:
            raise FileNotFoundError(filename)
        layer = vectors.GetLayer()
        if where_filter is not None:
            layer.SetAttributeFilter(where_filter)

        estimated_datatype = _validate_burn_value(burn_value, layer)
        if datatype is None:
            datatype = estimated_datatype

        vector_layer = VectorLayer(
            layer,
            scale,
            projection,
            name=filename,
            datatype=datatype,
            burn_value=burn_value,
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
        name: Optional[str] = None,
        datatype: int = gdal.GDT_Byte,
        burn_value: Union[int,float,str] = 1,
    ):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer
        self.name = name

        self._datatype = datatype

        # If the burn value is a number, use it directly, if it's a string
        # then assume it is a column name in the dataset
        self.burn_value = burn_value

        self._original = None

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
        area = Area(
            left=floor(min(x[0] for x in envelopes) / abs_xstep) * abs_xstep,
            top=ceil(max(x[3] for x in envelopes) / abs_ystep) * abs_ystep,
            right=ceil(max(x[1] for x in envelopes) / abs_xstep) * abs_xstep,
            bottom=floor(min(x[2] for x in envelopes) / abs_ystep) * abs_ystep,
        )

        super().__init__(area, scale, projection)


    @property
    def datatype(self) -> int:
        return self._datatype

    def read_array(self, xoffset, yoffset, xsize, ysize):
        # I did try recycling this object to save allocation/dealloction, but in practice it
        # seemed to only make things slower (particularly as you need to zero the memory each time yourself)
        dataset = gdal.GetDriverByName('mem').Create(
            'mem',
            xsize,
            ysize,
            1,
            self.datatype,
            []
        )
        if not dataset:
            raise MemoryError('Failed to create memory mask')

        dataset.SetProjection(self._projection)
        dataset.SetGeoTransform([
            self._active_area.left + (xoffset * self._pixel_scale.xstep),
            self._pixel_scale.xstep,
            0.0,
            self._active_area.top + (yoffset * self._pixel_scale.ystep),
            0.0,
            self._pixel_scale.ystep
        ])
        if isinstance(self.burn_value, (int, float)):
            gdal.RasterizeLayer(dataset, [1], self.layer, burn_values=[self.burn_value], options=["ALL_TOUCHED=TRUE"])
        elif isinstance(self.burn_value, str):
            gdal.RasterizeLayer(dataset, [1], self.layer, options=[f"ATTRIBUTE={self.burn_value}", "ALL_TOUCHED=TRUE"])
        else:
            raise ValueError("Burn value for layer should be number or field name")

        res = dataset.ReadAsArray(0, 0, xsize, ysize)
        return res
