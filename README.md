# Yirgacheffe: a declarative geospatial library for Python to make data-science with maps easier

## Overview

Yirgacheffe is an attempt to wrap raster and polygon geospatial datasets such that you can do computational work on them as a whole or at the pixel level, but without having to do a lot of the grunt work of working out where you need to be in rasters, or managing how much you can load into memory safely.

Example common use-cases:

* Do the datasets overlap? Yirgacheffe will let you define either the intersection or the union of a set of different datasets, scaling up or down the area as required.
* Rasterisation of vector layers: if you have a vector dataset then you can add that to your computation and yirgaceffe will rasterize it on demand, so you never need to store more data in memory than necessary.
* Do the raster layers get big and take up large amounts of memory? Yirgacheffe will let you do simple numerical operations with layers directly and then worry about the memory management behind the scenes for you.


## Installation

Yirgacheffe is available via pypi, so can be installed with pip for example:

```SystemShell
$ pip install yirgacheffe
```

## Basic usage

They main unit of data in Yirgacheffe is a "layer", which wraps either a raster dataset or polygon data, and then you can do work on layers without having to worry (unless you choose to) about how they align - Yirgacheffe will work out all the details around overlapping

The motivation for Yirgacheffe layers is to make working with gdal data slightly easier - it just hides some common operations, such as incremental loading to save memory, or letting you align layers to generate either the intersection result or the union result.

For example, if we wanted to do a simple [Area of Habitat](https://github.com/quantifyearth/aoh-calculator/) calculation, whereby we find the pixels where a species resides by combining its range polygon, its habitat preferences, and its elevation preferences, the code would be like this:

```python
import yirgaceffe as yg

habitat_map = yg.read_raster("habitats.tif")
elevation_map = yg.read_raster('elevation.tif')
range_polygon = yg.read_shape('species123.geojson')
area_per_pixel_map = yg.read_raster('area_per_pixel.tif')

refined_habitat = habitat_map.isin([...species habitat codes...])
refined_elevation = (elevation_map >= species_min) && (elevation_map <= species_max)

aoh = refined_habitat * refined_elevation * range_polygon * area_per_pixel_map

print(f'area for species 123: {aoh.sum()}')
```

Similarly, you could save the result to a new raster layer:

```python
...
aoh.to_geotiff("result.tif")
```

Yirgacheffe will automatically infer if you want to do an intersection of maps or a union of the maps based on the operators you use (see below for a full table). You can explicitly override that if you want.

### Lazy loading and evaluation

Yirgacheffe uses a technique from computer science called "lazy evaluation", which means that only when you resolve a calculation will Yirgacheffe do any work. So in the first code example given, the work is only calculated when you call the `sum()` method. All the other intermediary results such as `refined_habitat` and `refined_elevation` are not calculated either until that final `sum()` is called. You could easily call sum on those intermediaries if you wanted and get their results, and that would cause them to be evaluated then.

Similarly, when you load a layer, be it a raster layer or a vector layer from polygon data, the full data for the file isn't loaded until it's needed for a calculation, and even then only the part of the data necessary will be loaded or rasterized. Furthermore, Yirgacheffe will load the data in chunks, letting you work with rasters bigger than those that would otherwise fit within your computer's memory.


### Automatic expanding and contracting layers

When you load raster layers that aren't of equal geographic area (that is, they have a different origin, size, or both)then Yirgacheffe will do all the math internally to ensure that it aligns the pixels geospatially when doing calculations.

If size adjustments are needed, then Yirgacheffe will infer from the calculations you're doing if it needs to either crop or enlarge layers. For instance, if you're summing two rasters it'll expand them to be the union of their two areas before adding them, filling in the missing parts with zeros. If you're multiplying or doing a logical AND of pixels then it'll find the intersection between the two rasters (as areas missing in one would cause the other layer to result in zero anyway).

Whilst it is hoped that the default behaviour makes sense in most cases, we can't anticipate all usages, and so if you want to be explicit about the result of any maps you can specify it yourself.

For example, to tell Yirgacheffe to make a union of a set of layers you'd write:

```python
layers = [habitat_map, elevation_map, range_polygon]
union_area = YirgacheffeLayer.find_union(layers)
for layer in layers:
    layer.set_window_for_union(union_area)
```

There is a similar set of methods for using the intersection.

If you have set either the intersection window or union window on a layer and you wish to undo that restriction, then you can simply call `reset_window()` on the layer.

### Direct access to data

If doing per-layer operations isn't applicable for your application, you can read the pixel values for all layers (including VectorLayers) by calling `read_array` similarly to how you would for GDAL. The data you access will be relative to the specified window - that is, if you've called either `set_window_for_intersection` or `set_window_for_union` then `read_array` will be relative to that and Yirgacheffe will clip or expand the data with zero values as necessary.


## Layer types

Note that as part of the move to the next major release, 2.0, we are adding simpler ways to create layers. Not all of those have been implemented yet, which is why this section has some inconsistencies. However, given many of the common cases are already covered, we present the new 2.0 style methods (`read_raster` and similar) here so you can write cleaner code today rather than making people wait for the final 2.0 release.

### RasterLayer

This is your basic GDAL raster layer, which you load from a geotiff.

```python
from yirgaceffe.layers import RasterLayer

with RasterLayer.layer_from_file('test1.tif') as layer:
    total = layer.sum()
```

The new 2.0 way of doing this is:

```python
import yirgacheffe as yg

with yg.read_raster('test.tif') as layer:
    total = layer.sum()
```

You can also create empty layers ready for you to store results, either by taking the dimensions from an existing layer. In both these cases you can either provide a filename to which the data will be written, or if you do not provide a filename then the layer will only exist in memory - this will be more efficient if the layer is being used for intermediary results.

```python
with RasterLayer.empty_raster_layer_like(layer1, "results.tiff") as result:
    ...
```

Or you can specify the geographic area directly:

```python
with RasterLayer.empty_raster_layer(
    Area(left=-10.0, top=10.0, right=-5.0, bottom=5.0),
    PixelScale(0.005,-0.005),
    gdal.GDT_Float64,
    "results.tiff"
) as result:
    ...
```

You can also create a new layer that is a scaled version of an existing layer:

```python
with RasterLayer.layer_from_file('test1.tif') as source:
    scaled = RasterLayer.scaled_raster_from_raster(source, PixelScale(0.0001, -0.0001), 'scaled.tif')
```

If the data is from a GeoTIFF that has a nodata value specified, then pixel values with that specified nodata value in them will be converted to NaN. You can override that by providing `ignore_nodata=True` as an optional argument to `layer_from_file` (or with the new 2.0 API, `read_raster`). You can find out if a layer has a nodata value by accessing the `nodata` property - it is None if there is no such value.

### VectorLayer

This layer will load vector data and rasterize it on demand as part of a calculation - because it only rasterizes the data when needed, it is memory efficient.

Because it will be rasterized you need to specify the pixel scale and map projection to be used when rasterising the data, and the common way to do that is by using one of your other layers.

```python
from yirgaceffe import WGS_84_PROJECTION
from yirgaceffe.window import PixelScale
from yirgaceffe.layers import VectorLayer

with VectorLayer.layer_from_file('range.gpkg', PixelScale(0.001, -0.001), WGS_84_PROJECTION) as layer:
    ...
```

The new 2.0 way of doing this is, if you plan to use the vector layer in calculation with other raster layers that will have projection information:

```python
import yirgacheffe as yg

with yg.read_shape('range.gpkg') as layer:
    ...
```

Of if you plan to use the layer on its own and want to specify a rasterisation projection you can do:

```python
import yirgacheffe as yg

with yg.read_shape('range.gpkg', (yg.WGS_84_PROJECTION, (0.001, -0.001))) as layer:
    ...
```

### GroupLayer

You can combine several layers into one virtual layer to save you worrying about how to merge them if you don't want to manually add the layers together. Useful when you have tile sets for example. Any area not covered by a layer in the group will return zeros.

```python
tile1 = RasterLayer.layer_from_file('tile_N10_E10.tif')
tile2 = RasterLayer.layer_from_file('tile_N20_E10.tif')
all_tiles = GroupLayer([tile1, tile2])
```

If you provide tiles that overlap then they will be rendered in reverse one, so in the above example if tile1 and tile2 overlap, then in that region you'd get the data from tile1.

To save you specifying each layer, there is a convenience method to let you just load a set of TIFs by filename:

```python
with GroupLayer.layer_from_files(['tile_N10_E10.tif', 'tile_N20_E10.tif']) as all_tiles:
    ...
```

Or you can just specify a directory and it'll find the tifs in there (you can also add your own custom file filter too):

```python
with GroupLayer.layer_from_directory('.') as all_tiles:
    ...
```

The new 2.0 way of doing this is:

```python
import yirgacheffe as yg

with yg.read_rasters(['tile_N10_E10.tif', 'tile_N20_E10.tif']) as all_tiles:
    ...
```

If any of the layers have a `nodata` value specified, then any pixel with that value will be masked out to allow data from other layers to be visible.


### TiledGroupLayer

This is a specialisation of GroupLayer, which you can use if your layers are all the same size and form a grid, as is often the case with map tiles. In this case the rendering code can be optimised and this class is significantly faster that GroupLayer.

```python
tile1 = RasterLayer.layer_from_file('tile_N10_E10.tif')
tile2 = RasterLayer.layer_from_file('tile_N20_E10.tif')
all_tiles = TiledGroupLayer([tile1, tile2])
```

The new 2.0 way of doing this is:

```python
import yirgacheffe as yg

with yg.read_rasters(['tile_N10_E10.tif', 'tile_N20_E10.tif'], tiled=True) as all_tiles:
    ...
```

Notes:

* You can have missing tiles, and these will be filled in with zeros.
* You can have tiles that overlap, so long as they still conform to the rule that all tiles are the same size and on a grid.

### Constants

At times it is useful to have a fixed constant in an expression. Typically, similar to numpy, if an expression involving layers has a constant in, Yirgacheffe will apply that to all pixels in the equation without need for further elaboration:

```python
with yg.read_raster("some_data.tif") as layer:
    doubled_layer = layer * 2.0
    ...
```

This can be useful in tasks where you have an optional layer in your code. For example, here the code optionally loads an area-per-pixel layer, which if not present can just be substituted with a 1.0:

```python
try:
    area_layer = yg.read_raster('myarea.tiff')
except FileDoesNotExist:
    area_layer = 1.0
```

However, as with numpy, Python can not make the correct inference if the constant value is the first term in the equation. In that case you need to explicitly wrap the value with `constant` to help Python understand what is happening:

```python
with yg.read_raster("some_data.tif") as layer:
    result = yg.constant(1.0) / layer
```


### H3CellLayer

If you have H3 installed, you can generate a mask layer based on an H3 cell identifier, where pixels inside the cell will have a value of 1, and those outside will have a value of 0.

Becuase it will be rasterized you need to specify the pixel scale and map projection to be used when rasterising the data, and the common way to do that is by using one of your other layers.

```python
hex_cell_layer = H3CellLayer('88972eac11fffff', layer1.pixel_scale, layer1.projection)
```


### UniformAreaLayer

In certain calculations you find you have a layer where all the rows of data are the same - notably geotiffs that contain the area of a given pixel do this due to how conventional map projections work. It's hugely inefficient to load the full map into memory, so whilst you could just load them as `Layer` types, we recommend you do:

```python
with UniformAreaLayer('area.tiff') as layer:
    ....
```

Note that loading this data can still be very slow, due to how image compression works. So if you plan to use area.tiff more than once, we recommend use save an optimised version - this will do the slow uncompression once and then save a minimal file to speed up future processing:

```python
if not os.path.exists('yirgacheffe_area.tiff'):
    UniformAreaLayer.generate_narrow_area_projection('area.tiff', 'yirgacheffe_area.tiff')
area_layer = UniformAreaLayer('yirgacheffe_area.tiff')
```


## Supported operations on layers

Once you have two layers, you can perform numerical analysis on them similar to how numpy works:

### Add, subtract, multiple, divide

Pixel-wise addition, subtraction, multiplication or division (both true and floor division), and remainder. Either between arrays, or with constants:

```python
with RasterLayer.layer_from_file('test1.tif') as layer1:
    with RasterLayer.layer_from_file('test2.tif') as layer2:
        with RasterLayer.empty_raster_layer_like(layer1, 'result.tif') as result:
            calc = layer1 + layer2
            calc.save(result)
```

or

```python
with RasterLayer.layer_from_file('test1.tif') as layer1:
    with RasterLayer.empty_raster_layer_like(layer1, 'result.tif') as result:
        calc = layer1 * 42.0
        calc.save(result)
```


The new 2.0 way of doing these are:

```python
with yg.read_raster('test1.tif') as layer1:
    with yg.read_raster('test2.tif') as layer2:
        result = layer1 + layer2
        result.to_geotiff("result.tif")
```

or

```python
with yg.read_raster('test1.tif') as layer1:
    result = layer1 * 42.0
    result.to_geotiff("result.tif")
```

### Boolean testing

Testing for equality, less than, less than or equal, greater than, and greater than or equal are supported on layers, along with logical or and logical and, as per this example, where `elevation_upper` and `elevation_lower` are scalar values:

```
filtered_elevation = (min_elevation_map <= elevation_upper) & (max_elevation_map >= elevation_lower)
```

### Power

Pixel-wise raising to a constant power:

```python
with RasterLayer.layer_from_file('test1.tif') as layer1:
    with RasterLayer.empty_raster_layer_like(layer1, 'result.tif') as result:
        calc = layer1 ** 0.65
        calc.save(result)
```

### Log, Exp, Clip, etc.

The following math operators common to numpy and other libraries are currently supported:

* abs
* ceil
* clip
* exp
* exp2
* floor
* isin
* log
* log2
* log10
* maximum
* minimum
* nan_to_num
* round

Typically these can be invoked either on a layer as a method:

```python
calc = layer1.log10()
```

Or via the operators module, as it's sometimes nicer to do it this way when chaining together operations in a single expression:

```python
import yirgaceffe.operators as yo

calc = yo.log10(layer1 / layer2)
```

### 2D matrix convolution

To facilitate image processing algorithms you can supply a weight matrix to generate a processed image. Currently this support only works for square weight matrices of an odd size.

For example, to apply a blur function to a raster:

```python
blur_filter = np.array([
    [0.0, 0.1, 0.0],
    [0.1, 0.6, 0.1],
    [0.0, 0.1, 0.0],
])
with RasterLayer.layer_from_file('original.tif') as layer1:
    with RasterLayer.empty_raster_layer_like(layer1, 'blurred.tif') as result:
        calc = layer1.conv2d(blur_filter)
        calc.save(result)
```

### Type conversion

Similar to numpy and other Python numerical libraries, Yirgacheffe will automatically deal with simple type conversion where possible, however sometimes explicit conversion is either necessary or desired. Similar to numpy, there is an `astype` operator that lets you set the conversion:

```python
from yirgacheffe.operations import DataType


with RasterLayer.layer_from_file('float_data.tif') as float_layer:
    int_layer = float_layer.astype(DataType.Int32)
```

### Apply

You can specify a function that takes either data from one layer or from two layers, and returns the processed data. There's two version of this: one that lets you specify a numpy function that'll be applied to the layer data as an array, or one that is more shader like that lets you do pixel wise processing.

Firstly the numpy version looks like this:

```python
def is_over_ten(input_array):
    return numpy.where(input_array > 10.0, 0.0, 1.0)

layer1 = RasterLayer.layer_from_file('test1.tif')
result = RasterLayer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1.numpy_apply(is_over_ten)

calc.save(result)
```

or

```python
def simple_add(first_array, second_array):
    return first_array + second_array

layer1 = RasterLayer.layer_from_file('test1.tif')
layer2 = RasterLayer.layer_from_file('test2.tif')
result = RasterLayer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1.numpy_apply(simple_add, layer2)

calc.save(result)
```

If you want to do something specific on the pixel level, then you can also do that, again either on a unary or binary form.

```python
def is_over_ten(input_pixel):
    return 1.0 if input_pixel > 10 else 0.0

layer1 = RasterLayer.layer_from_file('test1.tif')
result = RasterLayer.empty_raster_layer_like(layer1, 'result.tif')

calc = layer1.shader_apply(is_over_ten)

calc.save(result)
```

Note that in general `numpy_apply` is considerably faster than `shader_apply`.

## Getting an answer out

There are two ways to store the result of a computation. In all the above examples we use the `save` call, to which you pass a gdal dataset band, into which the results will be written. You can optionally pass a callback to save which will be called for each chunk of data processed and give you the amount of progress made so far as a number between 0.0 and 1.0:

```python
def print_progress(p)
    print(f"We have made {p * 100} percent progress")

...

calc.save(result, callback=print_progress)
```


The alternative is to call `sum` which will give you a total:

```python
with (
    RasterLayer.layer_from_file(...) as area_layer,
    VectorLayer(...) as mask_layer
):
    intersection = RasterLayer.find_intersection([area_layer, mask_layer])
    area_layer.set_intersection_window(intersection)
    mask_layer.set_intersection_window(intersection)

    calc = area_layer * mask_layer

    total_area = calc.sum()
```

Similar to sum, you can also call `min` and `max` on a layer or calculation.

## Experimental

The following features are considered experimental - they have test cases to show them working in limited circumstances, but they've not yet been tested on a wide range of use cases. We hope that you will try them out and let us know how they work out.

### RescaledRasterLayer

The RescaledRasterLayer will take a GeoTIFF and do on demand rescaling in memory to get the layer to match other layers you're working on.

```python
with RasterLayer.layer_from_file("high_density_file.tif") as high_density:
    with RescaledRasterLayer.layer_from_file("low_density_file.tif", high_density.pixel_scale) as matched_density:

        # Normally this next line would fail with two RasterLayers as they ahve a different pixel density
        intersection = RasterLayer.find_intersection([high_density, matched_density])
        high_density.set_intersection_window(intersection)
        matched_density.set_intersection_window(intersection)

        calc = high_density * matched_density
        total = calc.sum()

```

### Parallel saving

There is a parallel version of save that can use multiple CPU cores at once to speed up work, that is added as an experimental feature for testing in our wider codebase, which will run concurrently the save over many threads.

```python
calc.parallel_save(result)
```

By default it will use as many CPU cores as are available, but if you want to limit that you can pass an extra argument to constrain that:

```python
calc.parallel_save(result, parallelism=4)
```

Because of the number of tricks that Python plays under the hood this feature needs a bunch of testing to let us remove the experimental flag, but in order to get that testing we need to put it out there! Hopefully in the next release we can remove the experimental warning.

## GPU support

Yirgacheffe has multiple backends, with more planned. Currently you can set the `YIRGACHEFFE_BACKEND` environmental variable to select which one to use. The default is `NUMPY`:

* NUMPY: CPU based calculation using [numpy](https://numpy.org/)
* MLX: Apple/Intel GPU support with CPU fallback based on [MLX](https://ml-explore.github.io/mlx/build/html/index.html)

Note that GPU isn't always faster than CPU - it very much depends on the workload, so testing your particular use-case is important.

## Thanks

Thanks to discussion and feedback from my colleagues, particularly Alison Eyres, Patrick Ferris, Amelia Holcomb, and Anil Madhavapeddy.

Inspired by the work of Daniele Baisero in his AoH library.
