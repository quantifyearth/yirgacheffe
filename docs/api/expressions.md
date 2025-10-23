# Expressions

The power of Yirgacheffe comes from being able to operate on geospatial data as if it was an elemental type. You can combine and work with layers without worrying about individual pixels or how different layers are aligned spatially. Assuming your data is in the same projection and pixel scale, you can just get on working. For example, here's how a simple [Area of Habitat](https://github.com/quantifyearth/aoh-calculator/) calculation might be implemented:

```python
import yirgaceffe as yg

with (
    yg.read_raster("habitats.tif") as habitat_map,
    yg.read_raster('elevation.tif') as elevation_map,
    yg.read_shape('species123.geojson') as range_polygon
):
    refined_habitat = habitat_map.isin([...species habitat codes...])
    refined_elevation = (elevation_map >= species_min) & (elevation_map <= species_max)
    aoh = refined_habitat * refined_elevation * range_polygon
    print(f'area for species 123: {aoh.sum()}')
    aoh.to_geotiff("result.tif")
```

## Operators

### Add, subtract, multiple, divide

Pixel-wise addition, subtraction, multiplication or division (both true and floor division), and remainder. Either between arrays, or with constants:

```python
with (
    yg.read_raster('test1.tif') as layer1,
    yg.read_raster('test2.tif') as layer2
):
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
with yg.read_raster('test1.tif') as layer1:
    calc = layer1 ** 0.65
    calc.save("result.tif")
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
with yg.read_raster('original.tif') as layer1:
    calc = layer1.conv2d(blur_filter)
    calc.to_geotiff("blurred.tif")
```

### Type conversion

Similar to numpy and other Python numerical libraries, Yirgacheffe will automatically deal with simple type conversion where possible, however sometimes explicit conversion is either necessary or desired. Similar to numpy, there is an `astype` operator that lets you set the conversion:

```python
from yirgacheffe.operations import DataType


with yg.read_raster('float_data.tif') as float_layer:
    int_layer = float_layer.astype(DataType.Int32)
```

### Apply

You can specify a function that takes either data from one layer or from two layers, and returns the processed data. There's two version of this: one that lets you specify a numpy function that'll be applied to the layer data as an array, or one that is more shader like that lets you do pixel wise processing.

Firstly the numpy version looks like this:

```python
def is_over_ten(input_array):
    return numpy.where(input_array > 10.0, 0.0, 1.0)

with yg.read_raster("test1.tif") as layer1:
    calc = layer1.numpy_apply(is_over_ten)
    calc.to_geotiff("result.tif")
```

or

```python
def simple_add(first_array, second_array):
    return first_array + second_array

with (
    yg.read_raster("test1.tif") as layer1,
    yg.read_raster("test2.tif") as layer2
):
    calc = layer1.numpy_apply(simple_add, layer2)
    calc.to_geotiff("result.tif")
```

If you want to do something specific on the pixel level, then you can also do that, again either on a unary or binary form.

```python
def is_over_ten(input_pixel):
    return 1.0 if input_pixel > 10 else 0.0

with yg.read_raster("test1.tif") as layer1:
    calc = layer1.shader_apply(is_over_ten)
    calc.to_geotiff(result)
```

Note:

1. Using `numpy_apply` prevents GPU optimisations occuring, so should be used as a last resort.
2. In general `numpy_apply` is considerably faster than `shader_apply`.

## Storing the results of expressions

There are three ways to store the result of a computation.

### Saving to a GeoTIFF

In all the above examples we use the `to_geotiff` call, to which you pass a filename for a GeoTIFF, into which the results will be written. You can optionally pass a callback to save which will be called for each chunk of data processed and give you the amount of progress made so far as a number between 0.0 and 1.0:

```python
def print_progress(p)
    print(f"We have made {p * 100} percent progress")

...

calc.to_geotiff(result, callback=print_progress)
```

### Aggregations

The alternative is to call aggregation functions such as `sum`, `min`, or `max` which will give you a single value by aggregating the data within the layer or expression:

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

### As numpy arrays

Finally, if you want to read the pixel values of either a layer or an expression, you can call `read_array`:

```python
import yirgacheffe as yg

with (
    yg.read_raster("test1.tif") as layer1,
    yg.read_raster("test2.tif") as layer2
):
    result = layer1 + layer2
    pixels = result.read_array(10, 10, 100, 100)
```
