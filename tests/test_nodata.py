import numpy as np

from yirgacheffe.layers.rasters import RasterLayer
from yirgacheffe.layers.group import GroupLayer

from tests.helpers import gdal_dataset_with_data

def test_raster_without_nodata_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 5.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    with RasterLayer(dataset) as layer:
        assert layer.nodata is None
        actual = layer.read_array(0, 0, 4, 2)
        assert np.array_equal(data1, actual, equal_nan=True)

def test_raster_with_nodata_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 5.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    dataset.GetRasterBand(1).SetNoDataValue(5.0)
    with RasterLayer(dataset) as layer:
        assert layer.nodata == 5.0
        data1[data1 == 5.0] = np.nan
        actual = layer.read_array(0, 0, 4, 2)
        assert np.array_equal(data1, actual, equal_nan=True)

def test_raster_with_nodata_value_ignored() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 5.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    dataset.GetRasterBand(1).SetNoDataValue(5.0)
    with RasterLayer(dataset, ignore_nodata=True) as layer:
        assert layer.nodata == 5.0
        actual = layer.read_array(0, 0, 4, 2)
        assert np.array_equal(data1, actual, equal_nan=True)

def test_group_layer_with_nodata_values() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    dataset1.GetRasterBand(1).SetNoDataValue(5.0)

    data2 = np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 6.0, 7.0, 8.0]])
    dataset2 = gdal_dataset_with_data((0.0, 0.0), 0.02, data2)
    dataset2.GetRasterBand(1).SetNoDataValue(1.0)

    with RasterLayer(dataset1) as layer1:
        with RasterLayer(dataset2) as layer2:
            with GroupLayer([layer1, layer2]) as group:
                actual = group.read_array(0, 0, 4, 2)
                expected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
                assert np.array_equal(expected, actual, equal_nan=True)

def test_group_layer_with_nodata_values_ignore_nodata() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    dataset1 = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    dataset1.GetRasterBand(1).SetNoDataValue(5.0)

    data2 = np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 6.0, 7.0, 8.0]])
    dataset2 = gdal_dataset_with_data((0.0, 0.0), 0.02, data2)
    dataset2.GetRasterBand(1).SetNoDataValue(1.0)

    with RasterLayer(dataset1, ignore_nodata=True) as layer1:
        with RasterLayer(dataset2, ignore_nodata=True) as layer2:
            with GroupLayer([layer1, layer2]) as group:
                actual = group.read_array(0, 0, 4, 2)
                assert np.array_equal(data1, actual, equal_nan=True)

def test_group_layer_with_nodata_read_from_empty_area() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    dataset1 = gdal_dataset_with_data((0.0, 10.0), 1.0, data1)
    dataset1.GetRasterBand(1).SetNoDataValue(5.0)

    data2 = np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 6.0, 7.0, 8.0]])
    dataset2 = gdal_dataset_with_data((0.0, -8.0), 1.0, data2)
    dataset2.GetRasterBand(1).SetNoDataValue(1.0)

    with RasterLayer(dataset1) as layer1:
        with RasterLayer(dataset2) as layer2:
            with GroupLayer([layer1, layer2]) as group:

                assert group.window.xsize == 4
                assert group.window.ysize == 20

                actual = group.read_array(0, 0, 4, 2)
                expected = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
                assert np.array_equal(expected, actual, equal_nan=True)

                actual = group.read_array(0, 10, 4, 2)
                expected = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
                assert np.array_equal(expected, actual, equal_nan=True)

                actual = group.read_array(0, 18, 4, 2)
                expected = np.array([[0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0]])
                assert np.array_equal(expected, actual, equal_nan=True)
