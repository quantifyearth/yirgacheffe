from __future__ import annotations
import os
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence

import numpy as np
from osgeo import gdal

from .layers import UniformAreaLayer
from .layers import YirgacheffeLayer
from .layers import ReprojectedRasterLayer, ResamplingMethod
from .layers import ConstantLayer
from .layers import GroupLayer, TiledGroupLayer
from .layers import RasterLayer
from .layers import VectorLayer
from .layers import AreaPerPixelLayer
from .window import MapProjection
from ._backends.enumeration import dtype as DataType

def read_raster(
    filename: Path | str,
    band: int = 1,
    ignore_nodata: bool = False,
) -> YirgacheffeLayer:
    """Open a raster file (e.g., GeoTIFF).

    Args:
        filename: Path of raster file to open.
        band: For multi-band rasters, which band to use (defaults to first if not specified).
        ignore_nodata: If the GeoTIFF has a NODATA value, don't substitute that value for NaN.

    Returns:
        An layer representing the raster data.

    Examples:
        >>> import yirgacheffe as yg
        >>> with yg.read_raster('test.tif') as layer:
        ...     total = layer.sum()
    """
    return RasterLayer.layer_from_file(filename, band, ignore_nodata)

def read_raster_like(
    filename: Path | str,
    like: YirgacheffeLayer,
    method: ResamplingMethod,
    band: int = 1,
    ignore_nodata: bool = False,
) -> YirgacheffeLayer:
    """Open a raster file but reproject it to match another layer.

    This method can be used to reproject a raster in one map projection to match another open layer.

    The reprojection will be only done when the data is actually used in a calculation, so unused data will
    not be reprojected.

    Args:
        filename: Path of raster file to open.
        like: Another layer that has a projection and pixel scale set. This layer will
            use the same projection and pixel scale as that one.
        method: The resampling method that will be used during the reprojection.
        band: For multi-band rasters, which band to use (defaults to first if not specified).
        ignore_nodata: If the GeoTIFF has a NODATA value, don't substitute that value for NaN.

    Returns:
        An layer representing the raster data in the new projection.

    Note:
        For parallel processing, use Python's multiprocessing module. This layer is safe when used
        with multiprocessing, but calls to `read_array` on an object are not thread safe.

    Examples:
        >>> import yirgacheffe as yg
        >>> with (
        ...     yg.read_raster('map_in_esri_54009.tif') as layer1,
        ...     yg.read_raster_like('map_in_wgs84.tif', layer1, yg.ResamplingMethod.Nearest) as layer2
        ... ):
        ...     res = layer1 * layer2
        ...     res.to_geotiff('result_in_esri_54009.tif')
    """
    if like.map_projection is None:
        raise ValueError("Reference layer must have a map projection.")
    original = RasterLayer.layer_from_file(filename, band, ignore_nodata)
    return ReprojectedRasterLayer(original, like.map_projection, method, original.name)

def read_narrow_raster(
    filename: Path | str,
    band: int = 1,
    ignore_nodata: bool = False,
) -> YirgacheffeLayer:
    """Open a 1 pixel wide raster file as a global raster.

    This exists for the special use case where an area per pixel raster would have the same value per horizontal row
    (e.g., a WGS84 map projection). For that case you can use this to load a raster that is 1 pixel wide and have
    it automatically expanded to act like a global raster in calculations.

    Args:
        filename: Path of raster file to open.
        band: For multi-band rasters, which band to use (defaults to first if not specified).
        ignore_nodata: If the GeoTIFF has a NODATA value, don't substitute that value for NaN.

    Returns:
        An layer representing the raster data.
    """
    return UniformAreaLayer.layer_from_file(filename, band, ignore_nodata)

def read_rasters(
    filenames : Sequence[Path | str],
    tiled: bool=False
) -> YirgacheffeLayer:
    """Open a set of raster files (e.g., GeoTIFFs) as a single layer.

    Args:
        filenames: List of paths of raster files to open.
        tiled: If you know that the rasters for a regular tileset, then setting this flag allows
            Yirgacheffe to perform certain optimisations that significantly improve performance for
            this use case.

    Returns:
        An layer representing the raster data.

    Examples:
        >>> import yirgacheffe as yg
        >>> with yg.read_rasters(['tile_N10_E10.tif', 'tile_N20_E10.tif']) as all_tiles:
        ...    ...
    """
    # Heading off a common error
    if isinstance(filenames, (str, Path)):
        raise TypeError(
            "Expected a sequence of files, not a single file. "
            f"Got {type(filenames).__name__}. Did you mean to pass [{filenames!r}]?"
        )
    if not tiled:
        return GroupLayer.layer_from_files(filenames)
    else:
        return TiledGroupLayer.layer_from_files(filenames)

def read_shape(
    filename: Path | str,
    projection: MapProjection | tuple[str, tuple[float, float]] | None = None,
    where_filter: str | None = None,
    datatype: DataType | None = None,
    burn_value: int | float | str = 1,
) -> YirgacheffeLayer:
    """Open a polygon file (e.g., GeoJSON, GPKG, or ESRI Shape File).

    Args:
        filename: Path of vector file to open.
        projection: The map projection to use.
        where_filter: For use with files with many entries (e.g., GPKG), applies this filter to the data.
        datatype: Specify the data type of the raster data generated.
        burn_value: The value of each pixel in the polygon.

    Returns:
        An layer representing the vector data.

    Examples:
        >>> import yirgacheffe as yg
        >>> with yg.read_shape('range.gpkg') as layer:
        ...    ...
    """

    if projection is not None:
        if not isinstance(projection, MapProjection):
            projection_name, scale_tuple = projection
            projection = MapProjection(projection_name, scale_tuple[0], scale_tuple[1])

    return VectorLayer._future_layer_from_file(
        filename,
        where_filter,
        projection,
        datatype,
        burn_value,
    )

def read_shape_like(
    filename: Path | str,
    like: YirgacheffeLayer,
    where_filter: str | None = None,
    datatype: DataType | None = None,
    burn_value: int | float | str = 1,
) -> YirgacheffeLayer:
    """Open a polygon file (e.g., GeoJSON, GPKG, or ESRI Shape File).

    Args:
        filename: Path of vector file to open.
        like: Another layer that has a projection and pixel scale set. This layer will
            use the same projection and pixel scale as that one.
        where_filter: For use with files with many entries (e.g., GPKG), applies this filter to the data.
        datatype: Specify the data type of the raster data generated.
        burn_value: The value of each pixel in the polygon.

    Returns:
        An layer representing the vector data.
    """
    return VectorLayer.layer_from_file_like(
        filename,
        like,
        where_filter,
        datatype,
        burn_value,
    )

def constant(value: int | float) -> YirgacheffeLayer:
    """Generate a layer that has the same value in all pixels regardless of scale, projection, and area.

    Generally this should not be necessary unless you must have the constant as the first term in an
    expression, as Yirgacheffe will automatically convert numbers into constant layers. However if the
    constant is the first term in the expression it must be wrapped by this call otherwise Python will
    not know that it should be part of the Yirgacheffe expression.

    Args:
        value: The value to be in each pixel of the expression term.

    Returns:
        A constant layer of the provided value.
    """
    return ConstantLayer(value)

def from_array(
    values: np.ndarray,
    origin: tuple[float, float],
    projection: MapProjection | tuple[str, tuple[float, float]],
) -> YirgacheffeLayer:
    """Creates an in-memory layer from a numerical array.

    Args:
        values: a 2D array of data values, with Y on the first dimension, X on
            the second dimension.
        origin: the position of the top left pixel in the geospatial space
        projection: the map projection and pixel scale to use.

    Returns:
        A geospatial layer that uses the provided data for its values.
    """

    if projection is None:
        raise ValueError("Projection must not be none")

    if not isinstance(projection, MapProjection):
        projection_name, scale_tuple = projection
        projection = MapProjection(projection_name, scale_tuple[0], scale_tuple[1])

    dims = values.shape

    # The original intent here was not to use GDAL directly, but use empty_raster_layer,
    # but then we end up tweaking the area multiple times, and hit rounding issues in
    # certain circumstances due to how it tries to align the output layer with a 0, 0
    # centered pixel grid.
    dataset = gdal.GetDriverByName('mem').Create(
        "mem",
        dims[1],
        dims[0],
        1,
        DataType.of_array(values).to_gdal(),
        [],
    )
    dataset.SetGeoTransform([
        origin[0], projection.xstep, 0.0, origin[1], 0.0, projection.ystep
    ])
    dataset.SetProjection(projection._gdal_projection)
    dataset.GetRasterBand(1).WriteArray(values, 0, 0)

    return RasterLayer(dataset)

def area_raster(
    projection: MapProjection | tuple[str, tuple[float, float]],
) -> YirgacheffeLayer:
    """Create a raster where the value of each pixel is the area in metres^2 of that pixel in the
    provided map projection and pixel scale.

    Args:
        projection: the map projection and pixel scale to use.

    Returns:
        A geospatial layer that where every pixel value is the geospatial area of that pixel.
    """

    if projection is None:
        raise ValueError("Projection must not be none")

    if not isinstance(projection, MapProjection):
        projection_name, scale_tuple = projection
        projection = MapProjection(projection_name, scale_tuple[0], scale_tuple[1])

    return AreaPerPixelLayer(projection)

def to_geotiff(
    filename: Path | str,
    bands: Sequence[YirgacheffeLayer],
    labels: list[str] | None = None,
    parallelism: int | bool | None = None,
    nodata: float | int | None = None,
    sparse: bool = False,
) -> None:
    """Save one or more results to a GeoTIFF file.

    Multiple results will be written as multiple bands in order. If labels are provided those will be
    assigned to the bands in the same order.

    All layers provided must be of the same map projection, pixel size, and data type. If necessary
    you can cast your layers using `astype` before passing them to this function.

    Args:
        filename: The name of the file to create.
        bands: A list of layers to store.
        labels: An optional list of band labels.
        parallelism: If passed, attempt to use multiple CPU cores up to the number provided, or if set to True,
            yirgacheffe will pick a sensible value.
        nodata: Nominate a value to be stored as nodata in the result.
        sparse: If True then save a sparse GeoTIFF as per GDAL's extension to the GeoTIFF standard.
    """
    if not bands:
        raise ValueError("Expected one or more layers to be written")

    if sparse and nodata is None:
        raise ValueError("Nodata value must be provided for sparse GeoTIFFs")

    layer_list = list(bands)
    first_layer = layer_list[0]
    for layer in layer_list[1:]:
        if layer.map_projection != first_layer.map_projection:
            raise ValueError("All layers must have the same map projection")
        if layer.datatype != first_layer.datatype:
            raise TypeError("All layers must have same data type. Use astype to explicitly cast layers.")

    if labels is not None and len(layer_list) != len(labels):
        raise ValueError("If labels are provided there must be the same number as there are layers")

    typed_filename = Path(filename)

    # We want to write to a tempfile before we move the result into place, but we can't use
    # the actual $TMPDIR as that might be on a different device, and so we use a file next to where
    # the final file will be, so we just need to rename the file at the end, not move it. But for special cases,
    # like GDAL's vsimem system we should not do this at all.
    target_dir = typed_filename.parent
    target_dir_parts = target_dir.parts
    is_vsi_based = len(target_dir_parts) == 2 and target_dir_parts[0] =='/' and target_dir_parts[1] in [
        'vsimem', 'vsizip', 'vsigzip', 'vsi7z', 'vsirar', 'vsitar', 'vsistdin',
        'vsistdout', 'vsisubfile', 'vsiparse', 'vsicached', 'vsicrypt',
    ]

    # This is a bit icky whilst set_window_for... is a public API, but it shouldn't really be in use
    union_area = YirgacheffeLayer.find_union(layer_list)
    for layer in layer_list:
        layer.set_window_for_union(union_area)

    # GDAL TIFF compression is a significant bottleneck, and threading really helps. Yirgacheffe otherwise
    # will parallel compute a block of data, and then wait as a single thread does the GDAL TIFF compression
    # which is very, very slow without multithreading. Because they operate in turn, we can give all the cores
    # to GDAL as when it's active we're not using them and vice versa.
    gdal_tiff_threads = None
    if parallelism:
        if isinstance(parallelism, bool):
            gdal_tiff_threads = cpu_count()
        else:
            gdal_tiff_threads = parallelism

    if not is_vsi_based:
        os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tempory_file:
        with RasterLayer.empty_raster_layer_like(
            first_layer,
            filename=tempory_file.name,
            nodata=nodata,
            sparse=sparse,
            threads=gdal_tiff_threads,
            bands=len(layer_list),
        ) as output:
            for index, layer in enumerate(layer_list):
                if parallelism is None:
                    _ = layer.save(output, band=index + 1)
                else:
                    if isinstance(parallelism, bool):
                        # Parallel save treats None as "work it out"
                        parallelism = None
                    _ = layer.parallel_save(output, parallelism=parallelism, band=index + 1)
                if labels:
                    output._dataset.GetRasterBand(index + 1).SetDescription(labels[index])

        if not is_vsi_based:
            os.rename(src=tempory_file.name, dst=typed_filename)
