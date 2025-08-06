import numpy as np

from yirgacheffe.layers.rasters import RasterLayer

from helpers import gdal_dataset_of_region, gdal_dataset_with_data

def test_raster_without_nodata_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 5.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    with RasterLayer(dataset) as layer:
        assert layer.nodata is None
        assert layer.sum() == data1.sum()

def test_raster_with_nodata_value() -> None:
    data1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 5.0, 8.0]])
    dataset = gdal_dataset_with_data((0.0, 0.0), 0.02, data1)
    dataset.GetRasterBand(1).SetNoDataValue(5.0)
    with RasterLayer(dataset) as layer:
        assert layer.nodata == 5.0
        # assert layer.sum() == (data1.sum() - 10)
