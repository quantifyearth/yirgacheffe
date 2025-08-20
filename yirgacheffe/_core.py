from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .layers.area import UniformAreaLayer
from .layers.base import YirgacheffeLayer
from .layers.constant import ConstantLayer
from .layers.group import GroupLayer, TiledGroupLayer
from .layers.rasters import RasterLayer
from .layers.vectors import VectorLayer
from .window import MapProjection
from ._backends.enumeration import dtype as DataType

def read_raster(
    filename: Union[Path,str],
    band: int = 1,
    ignore_nodata: bool = False,
) -> RasterLayer:
    """Open a raster file (e.g., GeoTIFF).

    Parameters
    ----------
    filename : Path
        Path of raster file to open.
    band : int, default=1
        For multi-band rasters, which band to use (defaults to first if not specified)
    ignore_nodata : bool, default=False
        If the GeoTIFF has a NODATA value, don't subsitute that value for NaN

    Returns
    -------
    RasterLayer
        Returns an layer representing the raster data.
    """
    return RasterLayer.layer_from_file(filename, band, ignore_nodata)

def read_narrow_raster(
    filename: Union[Path,str],
    band: int = 1,
    ignore_nodata: bool = False,
) -> RasterLayer:
    """Open a 1 pixel wide raster file as a global raster.

    This exists for the special use case where an area per pixel raster would have the same value per horizontal row
    (e.g., a WGS84 map projection). For that case you can use this to load a raster that is 1 pixel wide and have
    it automatically expanded to act like a global raster in calculations.

    Parameters
    ----------
    filename : Path
        Path of raster file to open.
    band : int, default=1
        For multi-band rasters, which band to use (defaults to first if not specified)
    ignore_nodata : bool, default=False
        If the GeoTIFF has a NODATA value, don't subsitute that value for NaN

    Returns
    -------
    RasterLayer
        Returns an layer representing the raster data.
    """
    return UniformAreaLayer.layer_from_file(filename, band, ignore_nodata)

def read_rasters(
    filenames : Sequence[Union[Path,str]],
    tiled: bool=False
) -> GroupLayer:
    """Open a set of raster files (e.g., GeoTIFFs) as a single layer.

    Parameters
    ----------
    filenames : List[Path]
        List of paths of raster files to open.
    tiled : bool, default=False
        If you know that the rasters for a regular tileset, then setting this flag allows
        Yirgacheffe to perform certain optimisations that significantly improve performance for
        this use case.

    Returns
    -------
    GroupLayer
        Returns an layer representing the raster data.
    """
    if not tiled:
        return GroupLayer.layer_from_files(filenames)
    else:
        return TiledGroupLayer.layer_from_files(filenames)

def read_shape(
    filename: Union[Path,str],
    projection: Union[Optional[MapProjection],Optional[Tuple[str,Tuple[float,float]]]]=None,
    where_filter: Optional[str] = None,
    datatype: Optional[DataType] = None,
    burn_value: Union[int,float,str] = 1,
) -> VectorLayer:
    """Open a polygon file (e.g., GeoJSON, GPKG, or ESRI Shape File).

    Parameters
    ----------
    filename : Path
        Path of raster file to open.
    projection: MapProjection or tuple, optional
        The map projection to use,
    where_filter : str, optional
        For use with files with many entries (e.g., GPKG), applies this filter to the data.
    datatype: DataType, default=DataType.Byte
        Specify the data type of the raster data generated.
    burn_value: int or float or str, default=1
        The value of each pixel in the polygon.

    Returns
    -------
    VectorLayer
        Returns an layer representing the vector data.
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
    filename: Union[Path,str],
    like: YirgacheffeLayer,
    where_filter: Optional[str] = None,
    datatype: Optional[DataType] = None,
    burn_value: Union[int,float,str] = 1,
) -> VectorLayer:
    """Open a polygon file (e.g., GeoJSON, GPKG, or ESRI Shape File).

    Parameters
    ----------
    filename : Path
        Path of raster file to open.
    like: YirgacheffeLayer
        Another layer that has a projection and pixel scale set. This layer will
        use the same projection and pixel scale as that one.
    where_filter : str, optional
        For use with files with many entries (e.g., GPKG), applies this filter to the data.
    datatype: DataType, default=DataType.Byte
        Specify the data type of the raster data generated.
    burn_value: int or float or str, default=1
        The value of each pixel in the polygon.

    Returns
    -------
    VectorLayer
        Returns an layer representing the vector data.
    """
    return VectorLayer.layer_from_file_like(
        filename,
        like,
        where_filter,
        datatype,
        burn_value,
    )

def constant(value: Union[int,float]) -> ConstantLayer:
    """Generate a layer that has the same value in all pixels regardless of scale, projection, and area.

    Generally this should not be necessary unless you must have the constant as the first term in an
    expression, as Yirgacheffe will automatically convert numbers into constant layers. However if the
    constant is the first term in the expression it must be wrapped by this call otherwise Python will
    not know that it should be part of the Yirgacheffe expression.

    Parameters
    ----------
    value : int or float
        The value to be in each pixel of the expression term.

    Returns
    -------
    ConstantLayer
        Returns a constant layer of the provided value.
    """
    return ConstantLayer(value)
