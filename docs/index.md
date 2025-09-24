# Introduction

<img src="assets/yirgacheffe-banner.svg" alt="Description" width="1020" height="510">

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

* `NUMPY`: CPU based calculation using [numpy](https://numpy.org/)
* `MLX`: Apple/Intel GPU support with CPU fallback based on [MLX](https://ml-explore.github.io/mlx/build/html/index.html)

Note that GPU isn't always faster than CPU - it very much depends on the workload, so testing your particular use-case is important.

## About

Yirgacheffe was created by [Michael Dales](https://digitalflapjack.com/) whilst working on multiple geospatial projects as a way to hide the boilerplate associated with working with large raster and polygon geospatial datasets.

The code for Yirgcheffe is [available on Github](https://github.com/quantifyearth/yirgacheffe/).
