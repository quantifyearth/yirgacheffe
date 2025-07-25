from pathlib import Path
from typing import Optional, Tuple, Union

from .layers import RasterLayer, VectorLayer, YirgacheffeLayer
from .window import PixelScale
from .operators import DataType

def read_raster(filename: Union[Path,str], band: int = 1) -> RasterLayer:
    return RasterLayer.layer_from_file(filename, band)

def read_shape(
    filename: Union[Path,str],
    where_filter: Optional[str],
    scale: PixelScale,
    projection: str,
    datatype: Optional[Union[int, DataType]] = None,
    burn_value: Union[int,float,str] = 1,
    anchor: Tuple[float,float] = (0.0, 0.0)
) -> VectorLayer:
    return VectorLayer.layer_from_file(
        filename,
        where_filter,
        scale,
        projection,
        datatype,
        burn_value,
        anchor
    )

def read_shape_like(
    filename: Union[Path,str],
    like: YirgacheffeLayer,
    where_filter: Optional[str],
    datatype: Optional[DataType] = None,
    burn_value: Union[int,float,str] = 1,
) -> VectorLayer:
    return VectorLayer.layer_from_file_like(
        filename,
        like,
        where_filter,
        datatype,
        burn_value,
    )
