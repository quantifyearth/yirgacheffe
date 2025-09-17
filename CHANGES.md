
## v1.7.8 (17/9/2025)

### Added

* Added marker for mypy that Yirgacheffe has type annotations

## v1.7.7 (8/9/2025)

### Added

* Automatically set rlimit for NOFILES so that parallel operations of machines with many cores don't run out of file descriptors.

### Removed

* Removed internal classes `LayerOperation`, `LayerMathMixin`, and `LayerConstant` from public interface.

## v1.7.6 (20/8/2025)

### Fixed

* Fixed issue whereby vector layers without explicit projection would use the abstract rather than concrete area value when generating a target raster.

### Added

* Added a core wrapper `constant` to save people using `ConstantLayer` explicitly
* Added a core wrapper `read_narrow_raster` to save people using `UniformAreaLayer` explicitly

## v1.7.5 (19/8/2025)

### Changed

* Minor improvements to GitHub Actions workflows

## v1.7.4 (19/8/2025)

### Fixed

* Fixed bug whereby reads from within a single tile that has nodata values in a group layer used the wrong numpy call to check for nan.

### Added

* Added `isnan` operator.

## v1.7.3 (18/8/2025)

### Fixed

* Fixed an issue introduced in 1.7.0 where `find_intersection` and `find_union` used the raw, non-pixel aligned area envelope.

## v1.7.2 (14/8/2025)

### Changed

* Added the option to set `parallelism=True` rather than just a number when calling `to_geotiff`, allowing Yirgacheffe to select the number of CPU cores to use for parallel operations.

## v1.7.1 (14/8/2025)

### Fixed

* Fixed an issue whereby if you used the MLX backend and called `read_array` the return value was sometimes an mlx array rather than a numpy array.

## v1.7 (14/8/2025)

### Added

* Support the ability to create VectorLayers that don't have a pixel scale or projection added. These layers will have the correct pixel scale and projection calculated when calculations on layers are saved or aggregated based on the other raster layers used in the calculation.
* Added MapProjection object to replace PixelScale objects and projection strings being separate parameters.
