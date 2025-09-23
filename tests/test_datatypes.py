
import numpy as np
import pytest
from osgeo import gdal

from yirgacheffe.operators import DataType
from yirgacheffe._backends import backend, BACKEND
from yirgacheffe.layers import RasterLayer

from tests.helpers import gdal_dataset_with_data

@pytest.mark.parametrize("gtype", [
    gdal.GDT_Int8,
    gdal.GDT_Int16,
    gdal.GDT_Int32,
    gdal.GDT_Int64,
    gdal.GDT_Byte,
    gdal.GDT_UInt16,
    gdal.GDT_UInt32,
    gdal.GDT_UInt64,
    gdal.GDT_Float32,
])
def test_round_trip(gtype) -> None:
    ytype = DataType.of_gdal(gtype)
    backend_type = backend.dtype_to_backend(ytype)
    assert backend.backend_to_dtype(backend_type) == ytype

@pytest.mark.parametrize("ytype", [
    DataType.Int8,
    DataType.Int16,
    DataType.Int32,
    DataType.Int64,
    DataType.UInt8,
    DataType.UInt16,
    DataType.UInt32,
    DataType.UInt64,
    DataType.Float32,
    DataType.Float64,
])
def test_round_trip_from_gdal(ytype) -> None:
    gtype = ytype.to_gdal()
    assert DataType.of_gdal(gtype) == ytype

def test_round_trip_float64() -> None:
    backend_type = backend.dtype_to_backend(DataType.Float64)
    ytype = backend.backend_to_dtype(backend_type)
    match BACKEND:
        case "NUMPY":
            assert ytype == DataType.Float64
        case "MLX":
            assert ytype == DataType.Float32

def test_float_to_int() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.5, 6.5, 7.5, 8.5]])

    layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
    # Note the float 32 here is to rule out that writing the result to the
    # new dataset was what caused the truncation
    result = RasterLayer.empty_raster_layer_like(layer1, datatype=DataType.Float32)

    comp = layer1.astype(DataType.UInt8)
    comp.save(result)

    expected = backend.promote(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
    actual = backend.demote_array(result.read_array(0, 0, 4, 2))
    assert (expected == actual).all()
