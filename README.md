# Yirgacheffe: a gdal wrapper that does the tricky bits

## Overview

Yirgacheffe is an attempt to wrap gdal datasets such that you can do computational work on them without having to worry about common tasks:

* Do the datasets overlap? Yirgacheffe will let you define either the intersection or the union of a set of different datasets, scaling up or down the area as required.
* Rasterisation of vector layers: if you have a vector dataset then you can add that to your computation and yirgaceffe will rasterize it on demand, so you never need to store more data in memory than necessary.

## Thanks

Inspired by the work of Daniele Baisero in his AoH library.