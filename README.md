# Yirgacheffe: a declarative geospatial library for Python to make data-science with maps easier

[![CI](https://github.com/quantifyearth/yirgacheffe/actions/workflows/pull-request.yml/badge.svg?branch=main)](https://github.com/quantifyearth/yirgacheffe/actions)
[![Documentation](https://img.shields.io/badge/docs-yirgacheffe.org-blue)](https://yirgacheffe.org)
[![PyPI version](https://img.shields.io/pypi/v/yirgacheffe)](https://pypi.org/project/yirgacheffe/)


## Overview

Yirgacheffe is a declarative geospatial library, allowing you to operate on both raster and polygon geospatial datasets without having to do all the tedious book keeping around layer alignment or dealing with hardware concerns around memory or parallelism. you can load into memory safely.

Example common use-cases:

* Do the datasets overlap? Yirgacheffe will let you define either the intersection or the union of a set of different datasets, scaling up or down the area as required.
* Rasterisation of vector layers: if you have a vector dataset then you can add that to your computation and yirgaceffe will rasterize it on demand, so you never need to store more data in memory than necessary.
* Do the raster layers get big and take up large amounts of memory? Yirgacheffe will let you do simple numerical operations with layers directly and then worry about the memory management behind the scenes for you.
* Parallelisation of operations over many CPU cores.
* Built in support for optionally using GPUs via [MLX](https://ml-explore.github.io/mlx/build/html/index.html) support.

## Installation

Yirgacheffe is available via pypi, so can be installed with pip for example:

```SystemShell
$ pip install yirgacheffe
```

## Documentation

The documentation can be found on [yirgacheffe.org](https://yirgacheffe.org/)

## Simple examples:

Here is how to do cloud removal from [Sentinel-2 data](https://browser.dataspace.copernicus.eu/?zoom=14&lat=6.15468&lng=38.20581&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1944lrmeTJcaSsnoxNMp4oucN1AjklGUANHd2cRZWyXnepHvzpaOWzMhH8SrWQo%2BqrOvOnu6f9FeCMrS%2FDZmvjzID%2FoE1tbOCEHK8ohPXjFqYojeR9%2B82ri&datasetId=S2_L2A_CDAS&fromTime=2025-09-09T00%3A00%3A00.000Z&toTime=2025-09-09T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE), using the [Scene Classification Layer](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/) data:

```python
import yirgaceffe as yg

with (
  yg.read_raster("T37NCG_20250909T073609_B06_20m.jp2") as vre2,
  yg.read_raster("T37NCG_20250909T073609_SCL_20m.jp2") as scl,
):
  is_cloud = (scl == 8) | (scl == 9) | (scl == 10)  # various cloud types
  is_shadow = (scl == 3)
  is_bad = is_cloud | is_shadow

  masked_vre2 = yg.where(is_bad, float("nan"), vre2)
  masked_vre2.to_geotiff("vre2_cleaned.tif")
```

or a species' [Area of Habitat](https://www.sciencedirect.com/science/article/pii/S0169534719301892) calculation:

```python
import yirgaceffe as yg

with (
    yg.read_raster("habitats.tif") as habitat_map,
    yg.read_raster('elevation.tif') as elevation_map,
    yg.read_shape('species123.geojson') as range_map,
):
    refined_habitat = habitat_map.isin([...species habitat codes...])
    refined_elevation = (elevation_map >= species_min) && (elevation_map <= species_max)
    aoh = refined_habitat * refined_elevation * range_polygon * area_per_pixel_map
    print(f'Area of habitat: {aoh.sum()}')
```

## Thanks

Thanks to discussion and feedback from my colleagues, particularly Alison Eyres, Patrick Ferris, Amelia Holcomb, and Anil Madhavapeddy.

Inspired by the work of Daniele Baisero in his AoH library.
