import numpy

from helpers import gdal_dataset_with_data
from yirgacheffe.layers import Layer


def test_sum_sans_window_update() -> None:
	data1 = numpy.array([[1, 2, 3, 4,], [5, 6, 7, 8,], [9, 10, 11, 12,], [13, 14, 15, 16,]])
	layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))
	assert layer1.sum() == numpy.sum(data1)

def test_sum_with_union() -> None:
	data1 = numpy.array([[1, 2, 3, 4,], [5, 6, 7, 8,], [9, 10, 11, 12,], [13, 14, 15, 16,]])
	data2 = numpy.array([[10, 20,], [50, 60,]])

	layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))
	layer2 = Layer(gdal_dataset_with_data((1.0, -1.0), 1.0, data2))

	layers = [layer1, layer2]
	window = Layer.find_union(layers)
	for layer in layers:
		layer.set_window_for_union(window)

	comp = layer1 + layer2
	expected = numpy.array([[1, 2, 3, 4,], [5, 16, 27, 8,], [9, 60, 71, 12,], [13, 14, 15, 16,]])
	assert comp.sum() == numpy.sum(expected)

def test_sum_with_intersection() -> None:
	data1 = numpy.array([[1, 2, 3, 4,], [5, 6, 7, 8,], [9, 10, 11, 12,], [13, 14, 15, 16,]])
	data2 = numpy.array([[10, 20,], [50, 60,]])

	layer1 = Layer(gdal_dataset_with_data((0.0, 0.0), 1.0, data1))
	layer2 = Layer(gdal_dataset_with_data((1.0, -1.0), 1.0, data2))

	layers = [layer1, layer2]
	window = Layer.find_intersection(layers)
	for layer in layers:
		layer.set_window_for_intersection(window)

	comp = layer1 + layer2

	expected = numpy.array([[0, 0, 0, 0,], [0, 16, 27, 0,], [0, 60, 71, 0,], [0, 0, 0, 0,]])
	assert comp.sum() == numpy.sum(expected)
