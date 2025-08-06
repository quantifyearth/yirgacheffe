from pathlib import Path
from typing import List, Optional, Tuple, Union

from .layers.base import YirgacheffeLayer
from .layers.group import GroupLayer, TiledGroupLayer
from .layers.rasters import RasterLayer
from .layers.vectors import VectorLayer
from .window import PixelScale
from .operators import DataType

def read_raster(
    filename: Union[Path,str],
    band: int = 1
) -> RasterLayer:
    """Open a raster file (e.g., GeoTIFF).

    Parameters
    ----------
    filename : Path
        Path of raster file to open.
    band : int, default=1
        For multi-band rasters, which band to use (defaults to first if not specified).

    Returns
    -------
    RasterLayer
        Returns an layer representing the raster data.
    """
    return RasterLayer.layer_from_file(filename, band)

def read_rasters(
    filenames : Union[List[Path],List[str]],
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
    scale: Union[PixelScale, Tuple[float,float]],
    projection: str,
    where_filter: Optional[str] = None,
    datatype: Optional[DataType] = None,
    burn_value: Union[int,float,str] = 1,
) -> VectorLayer:
    """Open a polygon file (e.g., GeoJSON, GPKG, or ESRI Shape File).

    Parameters
    ----------
    filename : Path
        Path of raster file to open.
    scale: PixelScale or tuple of float
        The dimensions of each pixel.
    projection: str
        The map projection to use
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

    if not isinstance(scale, PixelScale):
        scale = PixelScale(scale[0], scale[1])

    return VectorLayer.layer_from_file(
        filename,
        where_filter,
        scale,
        projection,
        datatype,
        burn_value
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
