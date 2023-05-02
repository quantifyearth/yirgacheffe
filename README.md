# Yirgacheffe: a gdal wrapper that does the tricky bits

## Overview

Yirgacheffe is an attempt to wrap gdal datasets such that you can do computational work on them without having to worry about common tasks:

* Do the datasets overlap? Yirgacheffe will let you define either the intersection or the union of a set of different datasets, scaling up or down the area as required.
* Rasterisation of vector layers: if you have a vector dataset then you can add that to your computation and yirgaceffe will rasterize it on demand, so you never need to store more data in memory than necessary.
* Do the raster layers get big and take up large amounts of memory? Yirgacheffe will let you do simple numerical operations with layers directly and then worry about the memory management behind the scenes for you.


## Basic layer usage

The motivation for Yirgacheffe layers is to make working with gdal data slightly easier - it just hides some common operations, such as incremental loading to save memory, or letting you align layers to generate either the intersection result or the union result.

For example, say we had three layers that overlapped and we wanted to know the

```python
from yirgacheffe.layer import Layer, UniformAreaLayer

elevation_layer = Layer.layer_from_file('elecation.tiff')
area_layer = UniformAreaLayer('area.tiff')
validity_layer = Layer.layer_from_file('validity.tiff')

# Work out the common subsection of all these and apply it to the layers
intersection = Layer.find_intersection(elecation_layer, area_layer, validity_layer)
elevation_layer.set_window_for_intersection(intersection)
area_layer.set_window_for_intersection(intersection)
validity_layer.set_window_for_intersection(intersection)

# Create a layer that is just the size of the intersection to store the results
result = Layer.empty_raster_layer(
    intersection,
    area_layer.pixel_scale,
    area_layer.datatype,
    "result.tif",
    area_layer.projection
)

# Work out the area where the data is valid and over 3000ft
def is_munro(data):
    return numpy.where(data > 3000.0, 0.0, 1.0)
calc = validity_layer * area_layer * elevation_layer.numpy_apply(is_munro)

calc.save(result)
```

If you want the union then you can simply swap do:

```python
intersection = Layer.find_union(elecation_layer, area_layer, validity_layer)
elevation_layer.set_window_for_union(intersection)
area_layer.set_window_for_union(intersection)
validity_layer.set_window_for_union(intersection)
```

If you want to work on the data in a layer directly you can call `read_array`, as you would with gdal. The data you access will be relative to the specified window - that is, if you've called either `set_window_for_intersection` or `set_window_for_union` then `read_array` will be relative to that and Yirgacheffe will clip or expand the data with zero values as necessary.

If you have set either the intersection window or union window on a layer and you wish to undo that restriction, then you can simply call `reset_window()` on the layer.


### Todo but not supported

Yirgacheffe is work in progress, so things planned but not supported currently:

* Pixel scale adjustment - all raster layers must be provided at the same pixel scale currently
* A fold operation
* CUPY support
* Dispatching work across multiple CPUs



## Layer types

### Layer

This is your basic GDAL raster layer, which you load from a geotiff.

```python
layer1 = Layer.layer_from_file('test1.tif')
```

You can also create empty layers ready for you to store results, either by taking the dimensions from an existing layer. In both these cases you can either provide a filename to which the data will be written, or if you do not provide a filename then the layer will only exist in memory - this will be more efficient if the layer is being used for intermediary results.

```python
result = Layer.empty_raster_layer_like(layer1, "results.tiff")
```

Or you can specify the geographic area directly:

```python
result = Layer.empty_raster_layer(
    Area(left=-10.0, top=10.0, right=-5.0, bottom=5.0),
    PixelScale(0.005,-0.005),
    gdal.GDT_Float64,
    "results.tiff"
)
```


### DynamicVectorRangeLayer

This layer will load vector data and rasterize it on demand as part of a calculation - becase it only rasterizes the data when needed, it is memory efficient.

Becuase it will be rasterized you need to specify the pixel scale and map projection to be used when rasterising the data, and the common way to do that is by using one of your other layers.

```python
vector_layer = DynamicVectorRangeLayer('range.gpkg', 'id_no == 42', layer1.pixel_scale, layer1.projection)
```


### UniformAreaLayer

In certain calculations you find you have a layer where all the rows of data are the same - notably geotiffs that contain the area of a given pixel do this due to how conventional map projections work. It's hugely inefficient to load the full map into memory, so whilst you could just load them as `Layer` types, we recommend you do:

```
area_layer = UniformAreaLayer('area.tiff')
```

Note that loading this data can still be very slow, due to how image compression works. So if you plan to use area.tiff more than once, we recommend use save an optimised version - this will do the slow uncompression once and then save a minimal file to speed up future processing:

```python
if not os.path.exists('yirgacheffe_area.tiff'):
    UniformAreaLayer('area.tiff', 'yirgacheffe_area.tiff')
area_layer = UniformAreaLayer('yirgacheffe_area.tiff')
```


### ConstantLayer

This is there to simplify code when you have some optional layers. Rather than littering your code with checks, you can just use a constant layer, which can be included in calculations and will just return an fixed value as if it wasn't there. Useful with 0.0 or 1.0 for sum or multiplication null layers.

```python
try:
    area_layer = UniformAreaLayer('myarea.tiff')
except FileDoesNotExist:
    area_layer = ConstantLayer(0.0)
```


### H3CellLayer

If you have H3 installed, you can generate a mask layer based on an H3 cell identifier, where pixels inside the cell will have a value of 1, and those outside will have a value of 0.

Becuase it will be rasterized you need to specify the pixel scale and map projection to be used when rasterising the data, and the common way to do that is by using one of your other layers.

```python
hex_cell_layer = H3CellLayer('88972eac11fffff', layer1.pixel_scale, layer1.projection)
```


## Supported operations on layers

Once you have two layers, you can perform numberical analysis on them similar to how numpy works:

### Add, subtract, multiple, divide

Pixel-wise addition, subtraction, multiplication or division, either between arrays, or with constants:

```python
layer1 = Layer.layer_from_file('test1.tif')
layer2 = Layer.layer_from_file('test2.tif')
result = Layer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1 + layer2

calculation.save(result)
```

or

```python
layer1 = Layer.layer_from_file('test1.tif')
result = Layer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1 * 42.0

calc.save(result)
```

### Power

Pixel-wise raising to a constant power:

```python
layer1 = Layer.layer_from_file('test1.tif')
result = Layer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1 ** 0.65

calc.save(result)
```


### Apply

You can specify a function that takes either data from one layer or from two layers, and returns the processed data. There's two version of this: one that lets you specify a numpy function that'll be applied to the layer data as an array, or one that is more shader like that lets you do pixel wise processing.

Firstly the numpy version looks like this:

```python
def is_over_ten(input_array):
    return numpy.where(input_array > 10.0, 0.0, 1.0)

layer1 = Layer.layer_from_file('test1.tif')
result = Layer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1.numpy_apply(is_over_ten)

calc.save(result)
```

or

```python
def simple_add(first_array, second_array):
    return first_array + second_array

layer1 = Layer.layer_from_file('test1.tif')
layer2 = Layer.layer_from_file('test2.tif')
result = Layer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1.numpy_apply(simple_add, layer2)

calc.save(result)
```

If you want to do something specific on the pixel level, then you can also do that, again either on a unary or binary form.

```python
def is_over_ten(input_pixel):
    return 1.0 if input_pixel > 10 else 0.0

layer1 = Layer.layer_from_file('test1.tif')
result = Layer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1.shader_apply(is_over_ten)

calc.save(result)
```

## Getting an answer out

There are two ways to store the result of a computation. In all the above examples we use the `save` call, to which you pass a gdal dataset band, into which the results will be written.

The alternative is to call `sum` which will give you a total:

```python
area_layer = Layer.layer_from_file(...)
mask_layer = DynamicVectorRangeLayer(...)

intersection = Layer.find_intersection([area_layer, mask_layer])
area_layer.set_intersection_window(intersection)
mask_layer.set_intersection_window(intersection)

calc = area_layer * mask_layer

total_area = calc.sum()
```


## Thanks

Thanks to discussion and feedback from the 4C team, particularly Alison Eyres, Amelia Holcomb, and Anil Madhavapeddy.

Inspired by the work of Daniele Baisero in his AoH library.