from __future__ import annotations
from math import ceil, floor
from pathlib import Path
from typing import Any

import deprecation
from osgeo import gdal, ogr

from .. import __version__
from ..window import Area, MapProjection, PixelScale
from .base import YirgacheffeLayer
from .rasters import RasterLayer
from .._backends import backend
from .._backends.enumeration import dtype as DataType

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
    @deprecation.deprecated(
        deprecated_in="1.7",
        removed_in="2.0",
        current_version=__version__,
        details="Use `VectorLayer` instead."
    )
    def layer_from_file( # type: ignore[override] # pylint: disable=W0221
        cls,
        filename: Path | str,
        where_filter: str | None,
        scale: PixelScale,
        projection: str,
        datatype: int | DataType | None = None,
        burn_value: int | float | str = 1,
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

        map_projection = MapProjection(projection, scale.xstep, scale.ystep)

        vector_layer = RasteredVectorLayer(
            layer,
            map_projection,
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
        projection: MapProjection,
        datatype: int | DataType = DataType.Byte,
        burn_value: int | float | str = 1,
    ):
        if layer is None:
            raise ValueError('No layer provided')
        self.layer = layer

        self._original: Any | None = None

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
        abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)
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

        dataset.SetProjection(projection.name)
        dataset.SetGeoTransform([area.left, projection.xstep, 0.0, area.top, 0.0, projection.ystep])
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
        filename: Path | str,
        other_layer: YirgacheffeLayer,
        where_filter: str | None = None,
        datatype: int | DataType | None = None,
        burn_value: int | float | str = 1,
    ) -> VectorLayer:
        if other_layer is None:
            raise ValueError("like layer can not be None")
        map_projection = other_layer.map_projection
        if map_projection is None:
            raise ValueError("Reference layer must have projectione")

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
            map_projection,
            name=str(filename),
            datatype=datatype if datatype is not None else other_layer.datatype,
            burn_value=burn_value,
            anchor=(other_layer.area.left, other_layer.area.top),
        )

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        vector_layer._dataset_path = filename if isinstance(filename, Path) else Path(filename)
        vector_layer._filter = where_filter
        return vector_layer

    @classmethod
    def layer_from_file(
        cls,
        filename: Path | str,
        where_filter: str | None,
        scale: PixelScale | None,
        projection: str | None,
        datatype: int | DataType | None = None,
        burn_value: int | float | str = 1,
        anchor: tuple[float, float] = (0.0, 0.0)
    ) -> VectorLayer:
        # In 2.0 we need to remove this and migrate to the MapProjection version
        if (projection is None) ^ (scale is None):
            raise ValueError("Either both projection and scale must be provide, or neither")
        if projection is not None and scale is not None:
            map_projection = MapProjection(projection, scale.xstep, scale.ystep)
        else:
            map_projection = None
        return cls._future_layer_from_file(
            filename,
            where_filter,
            map_projection,
            datatype,
            burn_value,
            anchor
        )

    @classmethod
    def _future_layer_from_file(
        cls,
        filename: Path | str,
        where_filter: str | None,
        projection: MapProjection | None,
        datatype: int | DataType | None = None,
        burn_value: int | float | str = 1,
        anchor: tuple[float, float] = (0.0, 0.0)
    ) -> VectorLayer:
        try:
            vectors = ogr.Open(filename)
        except RuntimeError as exc:
            # With exceptions on GDAL now returns the wrong (IMHO) exception
            raise FileNotFoundError(filename) from exc
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
            projection,
            name=str(filename),
            datatype=datatype_arg,
            burn_value=burn_value,
            anchor=anchor
        )

        # this is a gross hack, but unless you hold open the original file, you'll get
        # a SIGSEGV when using the layers from it later, as some SWIG pointers outlive
        # the original object being around
        vector_layer._original = vectors
        vector_layer._dataset_path = filename if isinstance(filename, Path) else Path(filename)
        vector_layer._filter = where_filter
        return vector_layer

    def __init__(
        self,
        layer: ogr.Layer,
        projection: MapProjection | None,
        name: str | None = None,
        datatype: int | DataType = DataType.Byte,
        burn_value: int | float | str = 1,
        anchor: tuple[float, float] = (0.0, 0.0)
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
        self._dataset_path: Path | None = None
        self._filter: str | None = None
        self._anchor: tuple[float, float] = (0.0, 0.0)

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
        self._anchor = anchor
        self._envelopes = envelopes

        if projection is not None:
            # Get the area, but scale it to the pixel resolution that we're using. Note that
            # the pixel scale GDAL uses can have -ve values, but those will mess up the
            # ceil/floor math, so we use absolute versions when trying to round.
            abs_xstep, abs_ystep = abs(projection.xstep), abs(projection.ystep)

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
        else:
            # If we don't have  a projection just go with the idealised area
            area = Area(
                left=floor(min(x[0] for x in envelopes)),
                top=ceil(max(x[3] for x in envelopes)),
                right=ceil(max(x[1] for x in envelopes)),
                bottom=floor(min(x[2] for x in envelopes)),
            )

        super().__init__(area, projection)


    def _get_operation_area(self, projection: MapProjection | None = None) -> Area:
        if self._projection is not None and projection is not None and self._projection != projection:
            raise ValueError("Calculation projection does not match layer projection")

        target_projection = projection if projection is not None else self._projection

        if target_projection is None:
            if self._active_area is not None:
                return self._active_area
            else:
                return self._underlying_area
        else:
            # Get the area, but scale it to the pixel resolution that we're using. Note that
            # the pixel scale GDAL uses can have -ve values, but those will mess up the
            # ceil/floor math, so we use absolute versions when trying to round.
            abs_xstep, abs_ystep = abs(target_projection.xstep), abs(target_projection.ystep)

            # Lacking any other reference, we will make the raster align with
            # (0.0, 0.0), if sometimes we want to align with an existing raster, so if
            # an anchor is specified, ensure we use that as our pixel space alignment
            x_anchor = self._anchor[0]
            y_anchor = self._anchor[1]
            left_shift = x_anchor - abs_xstep
            right_shift = x_anchor
            top_shift = y_anchor
            bottom_shift = y_anchor - abs_ystep

            envelopes = self._envelopes
            return Area(
                left=(floor((min(x[0] for x in envelopes) - left_shift) / abs_xstep) * abs_xstep) + left_shift,
                top=(ceil((max(x[3] for x in envelopes) - top_shift) / abs_ystep) * abs_ystep) + top_shift,
                right=(ceil((max(x[1] for x in envelopes) - right_shift) / abs_xstep) * abs_xstep) + right_shift,
                bottom=(floor((min(x[2] for x in envelopes) - bottom_shift) / abs_ystep) * abs_ystep) + bottom_shift,
            )

    @property
    def area(self) -> Area:
        if self._active_area is not None:
            return self._active_area
        else:
            return self._get_operation_area()

    def __getstate__(self) -> object:
        # Only support pickling on file backed layers (ideally read only ones...)
        if self._dataset_path is None or not self._dataset_path.exists():
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

    def _read_array_for_area(
        self,
        target_area: Area,
        target_projection: MapProjection | None,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Any:
        projection = target_projection if target_projection is not None else self._projection
        assert projection is not None

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

        dataset.SetProjection(projection.name)
        dataset.SetGeoTransform([
            target_area.left + (x * projection.xstep),
            projection.xstep,
            0.0,
            target_area.top + (y * projection.ystep),
            0.0,
            projection.ystep
        ])
        if isinstance(self.burn_value, (int, float)):
            gdal.RasterizeLayer(dataset, [1], self.layer, burn_values=[self.burn_value], options=["ALL_TOUCHED=TRUE"])
        elif isinstance(self.burn_value, str):
            gdal.RasterizeLayer(dataset, [1], self.layer, options=[f"ATTRIBUTE={self.burn_value}", "ALL_TOUCHED=TRUE"])
        else:
            raise ValueError("Burn value for layer should be number or field name")

        res = backend.promote(dataset.ReadAsArray(0, 0, width, height))
        return res

    def _read_array_with_window(self, _x, _y, _width, _height, _window) -> Any:
        raise NotImplementedError("VectorLayer does not support windowed reading")

    def _read_array(self, x: int, y: int, width: int, height: int) -> Any:
        return self._read_array_for_area(self.area, None, x, y, width, height)

    def read_array(self, x: int, y: int, width: int, height: int) -> Any:
        res = self._read_array(x, y, width, height)
        return backend.demote_array(res)
