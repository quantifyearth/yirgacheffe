
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

@pytest.mark.parametrize("ytype,expected", [
    (DataType.Byte, 1),
    (DataType.Int8, 1),
    (DataType.Int16, 2),
    (DataType.Int32, 4),
    (DataType.Int64, 8),
    (DataType.UInt8, 1),
    (DataType.UInt16, 2),
    (DataType.UInt32, 4),
    (DataType.UInt64, 8),
    (DataType.Float32, 4),
    (DataType.Float64, 8),
])
def test_datatype_size(ytype,expected) -> None:
    actual = ytype.sizeof()
    assert expected == actual

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
    actual = result.read_array(0, 0, 4, 2)
    assert (expected == actual).all()

@pytest.mark.parametrize("array,expected_type", [
    (
        np.ones((2, 2)).astype(np.int8),
        DataType.Int8,
    ),
    (
        np.ones((2, 2)).astype(np.int16),
        DataType.Int16,
    ),
    (
        np.ones((2, 2)).astype(np.int32),
        DataType.Int32,
    ),
    (
        np.ones((2, 2)).astype(np.int64),
        DataType.Int64,
    ),
    (
        np.ones((2, 2)).astype(np.uint8),
        DataType.UInt8,
    ),
    (
        np.ones((2, 2)).astype(np.uint16),
        DataType.UInt16,
    ),
    (
        np.ones((2, 2)).astype(np.uint32),
        DataType.UInt32,
    ),
    (
        np.ones((2, 2)).astype(np.uint64),
        DataType.UInt64,
    ),
    (
        np.ones((2, 2)).astype(np.float32),
        DataType.Float32,
    ),
    (
        np.ones((2, 2)).astype(np.float64),
        DataType.Float64,
    ),
])
def test_of_Array(array: np.ndarray, expected_type: DataType) -> None:
    ytype = DataType.of_array(array)
    assert ytype == expected_type
