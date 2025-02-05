import numpy as np

from helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer

def test_add_windows() -> None:
	data1 = np.array([[1, 2], [3, 4]])
	data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

	layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
	layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

	assert layer1.area != layer2.area
	assert layer1.window != layer2.window

	calc = layer1 + layer2

	assert calc.area == layer2.area
	assert calc.window == layer2.window

def test_multiply_windows() -> None:
	data1 = np.array([[1, 2], [3, 4]])
	data2 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

	layer1 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data1))
	layer2 = RasterLayer(gdal_dataset_with_data((0.0, 0.0), 0.02, data2))

	assert layer1.area != layer2.area
	assert layer1.window != layer2.window

	calc = layer1 * layer2

	assert calc.area == layer1.area
	assert calc.window == layer1.window
