# Yirgacheffe: a gdal wrapper that does the tricky bits

## Overview

Yirgacheffe is an attempt to wrap gdal datasets such that you can do computational work on them without having to worry about common tasks:

* Do the datasets overlap? Yirgacheffe will let you define either the intersection or the union of a set of different datasets, scaling up or down the area as required.
* Rasterisation of vector layers: if you have a vector dataset then you can add that to your computation and yirgaceffe will rasterize it on demand, so you never need to store more data in memory than necessary.


## Basic layer usage

The motivation for Yirgacheffe layers is to make working with gdal data slightly easier - it just hides some common operations, such as incremental loading to save memory, or letting you align layers to generate either the intersection result or the union result.

For example, say we had three layers that overlapped and we wanted to know the

```
elevation_layer = Layer.layer_from_file('elecation.tiff')
area_layer = UniformAreaLayer('area.tiff')
validity_layer = Layer.layer_from_file('validity.tiff')

# Work out the common subsection of all these and apply it to the layers
intersection = Layer.find_intersection(elecation_layer, area_layer, validity_layer)
elevation_layer.set_window_for_intersection(intersection)
area_layer.set_window_for_intersection(intersection)
validity_layer.set_window_for_intersection(intersection)

# Work out the area where the data is valid and over 3000ft
def is_munro(data):
	return numpy.where(data > 3000.0, 0.0, 1.0)
result = validity_layer * area_layer * elevation_layer.apply(is_munro)

result_band = result_gdal_dataset.GetRasterBand(1)
result.save(result_band)
```

If you want the union then you can simply swap do:

```
intersection = Layer.find_union(elecation_layer, area_layer, validity_layer)
elevation_layer.set_window_for_union(intersection)
area_layer.set_window_for_union(intersection)
validity_layer.set_window_for_union(intersection)
```

If you want to work on the data in a layer directly you can call `read_array`, as you would with gdal. The data you access will be relative to the specified window - that is, if you've called either `set_window_for_intersection` or `set_window_for_union` then `read_array` will be relative to that and Yirgacheffe will clip or expand the data with zero values as necessary.


## Layer types

### Layer

This is your basic GDAL raster layer, which you load from a geotiff.

```
layer1 = Layer.layer_from_file('test1.tif')
```

### DynamicVectorRangeLayer

This layer will load vector data and rasterize it on demand as part of a calculation - becase it only rasterizes the data when needed, it is memory efficient.


### UniformAreaLayer

In certain calculations you find you have a layer where all the rows of data are the same - notably geotiffs that contain the area of a given pixel do this due to how conventional map projections work. It's hugely inefficient to load the full map into memory, so whilst you could just load them as `Layer` types, we recommend you do:

```
area_layer = UniformAreaLayer('area.tiff')
```

Note that loading this data can still be very slow, due to how image compression works. So if you plan to use area.tiff more than once, we recommend use save an optimised version - this will do the slow uncompression once and then save a minimal file to speed up future processing:

```
if not os.path.exists('yirgacheffe_area.tiff'):
	UniformAreaLayer('area.tiff', 'yirgacheffe_area.tiff')
area_layer = UniformAreaLayer('yirgacheffe_area.tiff')
```


### NullLayer

This is there to simplify code when you have some optional layers. Rather than littering your code with checks, you can just use a null layer, which can be included in calculations and will just return an identity value as if it wasn't there.

```
try:
	area_layer = UniformAreaLayer('myarea.tiff')
except FileDoesNotExist:
	area_layer = NullLayer()
```


## Supported operations on layers

Once you have two layers, you can perform numberical analysis on them similar to how numpy works:

### Add, subtract, multiple, divide

Pixel-wise addition, subtraction, multiplication or division, either between arrays, or with constants:

```
layer1 = Layer.layer_from_file('test1.tif')
layer2 = Layer.layer_from_file('test2.tif')

result = layer1 + layer2

result_band = result_gdal_dataset.GetRasterBand(1)
result.save(result_band)
```

or

```
layer1 = Layer.layer_from_file('test1.tif')

result = layer1 * 42.0

result_band = result_gdal_dataset.GetRasterBand(1)
result.save(result_band)
```

### Power

Pixel-wise raising to a constant power:

```
layer1 = Layer.layer_from_file('test1.tif')

result = layer1 ** 0.65

result_band = result_gdal_dataset.GetRasterBand(1)
result.save(result_band)
```


### Apply

You can specify a function that takes either data from one layer or from two layers, and returns the processed data.

```
def is_over_ten(input_array):
	return numpy.where(input_array > 10.0, 0.0, 1.0)

layer1 = Layer.layer_from_file('test1.tif')

result = layer1.apply(is_over_ten)

result_band = result_gdal_dataset.GetRasterBand(1)
result.save(result_band)
```

or

```
def simple_add(first_array, second_array):
	return first_array + second_array

layer1 = Layer.layer_from_file('test1.tif')
layer2 = Layer.layer_from_file('test2.tif')

result = layer1.apply(simple_add, layer2)

result_band = result_gdal_dataset.GetRasterBand(1)
result.save(result_band)
```

## Thanks

Thanks to discussion and feedback from the 4C team, particularly Alison Eyres, Amelia Holcomb, and Anil Madhavapeddy.

Inspired by the work of Daniele Baisero in his AoH library.