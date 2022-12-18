import cProfile
import pstats

import numpy as np
from osgeo import gdal

from yirgacheffe.layers import Layer, Area, PixelScale

USE_CUDA = False
try:
	import cupy as cp
	USE_CUDA = True
except ImportError:
	pass

def direct_gdal_sum_single_layer(data: Layer):
	'''This is the direct gdal equivalent, using the usual gdal trick
	of loading a line at a time'''
	dataset = data._dataset
	band = dataset.GetRasterBand(1)
	count = 0.0
	for y in range(dataset.RasterYSize):
		block = band.ReadAsArray(0, y, dataset.RasterXSize, 1)
		count += block.sum()
	assert count == 0.0

def yirgacheffe_cpu_sum_single_layer(data: Layer):
	'''Yirgacheffe has much simpler code, but is it at the expense of speed?'''
	count = data.sum()
	assert count == 0.0

def yirgacheffe_cuda_sum_single_layer(data: Layer):
	raise NotImplementedError("tbd")

def profile_single_layer():
	tests = [('gdal_chunked', direct_gdal_sum_single_layer), ('basic       ', yirgacheffe_cpu_sum_single_layer)]
	if USE_CUDA:
		tests.append(('cuda', yirgacheffe_cuda_sum_single_layer))

	area = Area(left=-1.0, top=2.0, right=1.0, bottom=-2.0)
	scale = PixelScale(0.0001, -0.0001)
	testdata = Layer.empty_raster_layer(area, scale, gdal.GDT_Float64)

	for test in tests:
		for i in range(3):
			profiler = cProfile.Profile()
			profiler.enable()
			test[1](testdata)
			profiler.disable()
			p = pstats.Stats(profiler)
			print(f'{test[0]}: run {i} took {p.total_tt} seconds')

	p.sort_stats(pstats.SortKey.TIME).print_stats(20)



def direct_gdal_sum_single_layer_and_const(data: Layer):
	'''This is the direct gdal equivalent, using the usual gdal trick
	of loading a line at a time'''
	dataset = data._dataset
	band = dataset.GetRasterBand(1)
	count = 0.0
	for y in range(dataset.RasterYSize):
		block = band.ReadAsArray(0, y, dataset.RasterXSize, 1)
		calc = block + 42.0
		count += calc.sum()
	assert count != 0.0

def yirgacheffe_cpu_sum_single_layer_and_const(data: Layer):
	'''Yirgacheffe has much simpler code, but is it at the expense of speed?'''
	calc = data + 42.0
	count = calc.sum()
	assert count != 0.0

def yirgacheffe_cuda_sum_single_layer_and_const(data: Layer):
	raise NotImplementedError("tbd")

def profile_single_layer_and_const():
	tests = [('gdal_chunked', direct_gdal_sum_single_layer_and_const), ('basic       ', yirgacheffe_cpu_sum_single_layer_and_const)]
	if USE_CUDA:
		tests.append(('cuda', yirgacheffe_cuda_sum_single_layer_and_const))

	area = Area(left=-1.0, top=2.0, right=1.0, bottom=-2.0)
	scale = PixelScale(0.0001, -0.0001)
	testdata = Layer.empty_raster_layer(area, scale, gdal.GDT_Float64)

	for test in tests:
		for i in range(3):
			profiler = cProfile.Profile()
			profiler.enable()
			test[1](testdata)
			profiler.disable()
			p = pstats.Stats(profiler)
			print(f'{test[0]}: run {i} took {p.total_tt} seconds')

	p.sort_stats(pstats.SortKey.TIME).print_stats(20)



if __name__ == "__main__":
	profile_single_layer()
	profile_single_layer_and_const()
