## v1.9.0 (26/9/2025)

### Added

* Added the ability to call `read_array` on expressions (before you could only call it on layers).

### Changed

* The first argument of MapProjection, the string defining the projection used is now validated with the pyproj library, and can be in any from the pyproj `from_string` takes: Well Known Text (WKT) or "epsg:4326" or "esri:54009" etc. The name function still returns the WKT representation for backwards compatibility.
* The functions `pixel_from_latlng` and `latlng_from_pixel` will work regardless of the underlying map projection.

## v1.8.1 (25/9/2025)

### Fixed

* Fixed issue whereby calling `set_window_for_intersection` would fail if the pixel alignment on a vector layer was rounded unfortunately.

### Changed

* More documentation updates.

## v1.8.0 (24/9/2025)

### Added

* Mkdocs based documentation.

### Changed

* Modernised type hints to use latest Python standards.

## v1.7.9 (23/9/2025)

### Fixed

* Fix type inference for expressions so that `to_geotiff` selects correct GeoTIFF file type to store results as.

### Changed

* Improved typing of methods that take a filename to use both Path and str.

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
