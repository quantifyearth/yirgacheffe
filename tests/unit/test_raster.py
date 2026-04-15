import math
import os
import tempfile

import numpy as np
import pytest
from osgeo import gdal

import yirgacheffe as yg
from tests.unit.helpers import (
    gdal_dataset_of_region,
    gdal_multiband_dataset_with_data,
    gdal_dataset_with_data,
)
from yirgacheffe import Area, MapProjection, DataType
from yirgacheffe._datatypes import Window
from yirgacheffe._layers import RasterLayer, InvalidRasterBand


# There is a lot of "del" in this file, due to a combination of gdal having no way
# to explicitly close a file beyond forcing the gdal object's deletion, and Windows
# getting upset that it tries to clear up the TemporaryDirectory and there's an open
# file handle within that directory.


def test_make_basic_layer() -> None:
    projection = MapProjection("epsg:4326", 0.02, -0.02)
    area = Area(-10, 10, 10, -10, projection)
    dataset = gdal_dataset_of_region(area, 0.02)

    # The context manager routines are just called on the base class, so ensure
    # they do call the GDAL close method, otherwise all that is for nothing
    original_ref = dataset.Close
    close_called = []

    def mocked_close():
        close_called.append(True)
        original_ref()

    dataset.Close = mocked_close

    with RasterLayer(dataset) as layer:
        assert layer.area == area
        assert layer.projection == projection
        assert layer.area.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
        assert layer.dimensions == (1000, 1000)
        assert layer._virtual_window == Window(0, 0, 1000, 1000)

    assert close_called


def test_layer_from_null() -> None:
    # Seems a petty test, but gdal doesn't throw exceptions
    # so you often get None datasets if you're not careful
    with pytest.raises(ValueError):
        with RasterLayer(None) as _layer:
            pass


def test_layer_from_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = RasterLayer.layer_from_file("this_file_does_not_exist.tif")


def test_open_file() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        projection = MapProjection("epsg:4326", 0.02, -0.02)
        area = Area(-10, 10, 10, -10, projection)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        dataset.Close()
        assert os.path.exists(path)
        with RasterLayer.layer_from_file(path) as layer:
            assert layer.area == area
            assert layer.projection == projection
            assert layer.area.geo_transform == (-10, 0.02, 0.0, 10, 0.0, -0.02)
            assert layer.dimensions == (1000, 1000)
            assert layer._virtual_window == Window(0, 0, 1000, 1000)
            del layer


def test_empty_layer_from_raster():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    empty = RasterLayer.empty_raster_layer_like(source)
    assert empty.projection == source.projection
    assert empty.dimensions == source.dimensions
    assert empty._virtual_window == source._virtual_window
    assert empty.datatype == source.datatype
    assert empty.area == source.area
    assert empty._dataset.GetRasterBand(1).GetNoDataValue() is None


@pytest.mark.parametrize("nodata", [0, 0.0, 2, 2.0])
def test_empty_layer_from_raster_with_no_data_value(nodata):
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    empty = RasterLayer.empty_raster_layer_like(source, nodata=nodata)
    assert empty.projection == source.projection
    assert empty.dimensions == source.dimensions
    assert empty._virtual_window == source._virtual_window
    assert empty.datatype == source.datatype
    assert empty.area == source.area
    assert empty._dataset.GetRasterBand(1).GetNoDataValue() == nodata


def test_empty_layer_from_raster_with_new_smaller_area():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    smaller_area = Area(-1, 1, 1, -1)
    empty = RasterLayer.empty_raster_layer_like(source, area=smaller_area)
    assert empty.projection == source.projection
    assert empty.dimensions == (100, 100)
    assert empty._virtual_window == Window(0, 0, 100, 100)
    assert empty.datatype == source.datatype
    expected_geotransform = (-1.0, 0.02, 0.0, 1.0, 0.0, -0.02)
    for i in range(6):
        assert math.isclose(empty.area.geo_transform[i], expected_geotransform[i])


def test_empty_layer_from_raster_new_datatype():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))
    assert source.datatype == DataType.Byte
    empty = RasterLayer.empty_raster_layer_like(source, datatype=gdal.GDT_Float64)
    assert empty.projection == source.projection
    assert empty.dimensions == source.dimensions
    assert empty._virtual_window == source._virtual_window
    assert empty.datatype == DataType.Float64


def test_empty_layer_from_raster_with_window():
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.02))

    clipped_source = source.as_area(Area(-1, 1, 1, -1, source.projection))
    assert clipped_source.dimensions < source.dimensions

    empty = RasterLayer.empty_raster_layer_like(clipped_source)
    assert empty.projection == source.projection
    assert empty.area == clipped_source.area
    assert empty.dimensions == clipped_source.dimensions


@pytest.mark.parametrize(
    "size,expect_success",
    [
        ((5, 5), True),
        ((5, 1), True),
        ((1, 5), True),
        ((5, 0), False),
        ((0, 5), False),
    ],
)
def test_read_array_size(size, expect_success):
    source = RasterLayer(gdal_dataset_of_region(Area(-10, 10, 10, -10), 0.2))
    assert source.area == Area(-10, 10, 10, -10, MapProjection("epsg:4326", 0.2, -0.2))
    assert source.dimensions == (20 / 0.2, 20 / 0.2)
    assert source._virtual_window == Window(0, 0, 20 / 0.2, 20 / 0.2)

    if expect_success:
        data = source.read_array(0, 0, size[0], size[1])
        assert data.shape == (size[1], size[0])
    else:
        with pytest.raises(ValueError):
            _ = source.read_array(0, 0, size[0], size[1])


def test_invalid_band() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "test.tif")
        area = Area(-10, 10, 10, -10)
        dataset = gdal_dataset_of_region(area, 0.02, filename=path)
        dataset.Close()
        assert os.path.exists(path)
        with pytest.raises(InvalidRasterBand):
            _ = RasterLayer.layer_from_file(path, band=42)


def test_multiband_raster() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    data2 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

    datas = [data1, data2]
    dataset = gdal_multiband_dataset_with_data((0.0, 0.0), 0.02, datas)
    layer1 = RasterLayer(dataset, band=1)
    layer2 = RasterLayer(dataset, band=2)

    layers = [layer1, layer2]
    for i in range(2):
        data = datas[i]
        layer = layers[i]
        actual = layer.read_array(0, 0, 4, 2)
        assert (data == actual).all()


def test_read_array_is_numpy():
    # This test will fail if we use say the MLX backend without casting it back to numpy
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data)
    with RasterLayer(dataset) as layer1:
        actual = layer1.read_array(0, 0, 4, 2).astype(int)
        expected = data.astype(int)
        assert (actual == expected).all


def test_cse_hash_of_geotiff() -> None:
    # We assume that on disk rasters are immutable
    with tempfile.TemporaryDirectory() as tempdir:
        area = Area(-10, 10, 10, -10)

        path1 = os.path.join(tempdir, "test1.tif")
        _ = gdal_dataset_of_region(area, 0.02, filename=path1)
        path2 = os.path.join(tempdir, "test2.tif")
        _ = gdal_dataset_of_region(area, 0.02, filename=path2)

        with (
            yg.read_raster(path1) as layer0,
            yg.read_raster(path1) as layer1,
            yg.read_raster(path2) as layer2,
        ):
            assert layer0._cse_hash == layer1._cse_hash
            assert layer1._cse_hash != layer2._cse_hash


def test_cse_hash_in_memory() -> None:
    # We can't consider in memory rasters the same without hashing the entire dataset, so they
    # should not match
    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    with (
        yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer0,
        yg.from_array(data1, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer1,
        yg.from_array(data2, (0, 0), ("epsg:4326", (1.0, -1.0))) as layer2,
    ):
        assert layer0._cse_hash != layer1._cse_hash
        assert layer1._cse_hash != layer2._cse_hash
