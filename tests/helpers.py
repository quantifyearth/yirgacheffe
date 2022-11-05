from math import ceil
from typing import Set, Optional, Tuple

import numpy
from osgeo import gdal, ogr

from yirgacheffe.layers import Area, Layer

WSG_84_PROJECTION = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'\
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'\
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],'\
    'AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

def gdal_dataset_of_region(area: Area, pixel_pitch: float, filename: Optional[str]=None) -> gdal.Dataset:
    if filename:
        driver = gdal.GetDriverByName('GTiff')
    else:
        driver = gdal.GetDriverByName('mem')
        filename = 'mem'
    dataset = driver.Create(
        filename,
        ceil((area.right - area.left) / pixel_pitch),
        ceil((area.top - area.bottom) / pixel_pitch),
        1,
        gdal.GDT_Byte,
        []
    )
    dataset.SetGeoTransform([
        area.left, pixel_pitch, 0.0, area.top, 0.0, pixel_pitch * -1.0
    ])
    dataset.SetProjection(WSG_84_PROJECTION)
    # the dataset isn't valid until you populate the data
    band = dataset.GetRasterBand(1)
    for yoffset in range(dataset.RasterYSize):
        band.WriteArray(numpy.array([[(yoffset % 256),] * dataset.RasterXSize]), 0, yoffset)
    return dataset

def gdal_dataset_of_layer(layer: Layer, filename: Optional[str]=None) -> gdal.Dataset:
    if filename:
        driver = gdal.GetDriverByName('GTiff')
    else:
        driver = gdal.GetDriverByName('mem')
        filename = 'mem'
    dataset = driver.Create(
        filename,
        layer.window.xsize,
        layer.window.ysize,
        1,
        gdal.GDT_Float32,
        []
    )
    dataset.SetGeoTransform(layer.geo_transform)
    dataset.SetProjection(layer.projection)
    return dataset

def gdal_dataset_with_data(origin: Tuple, pixel_pitch: float, data: numpy.array) -> gdal.Dataset:
    assert data.ndim == 2
    datatype = gdal.GDT_Byte
    if isinstance(data[0][0], float):
        datatype = gdal.GDT_Float64
    dataset = gdal.GetDriverByName('mem').Create(
        'mem',
        len(data[0]),
        len(data),
        1,
        datatype,
        []
    )
    dataset.SetGeoTransform([
        origin[0], pixel_pitch, 0.0, origin[1], 0.0, pixel_pitch * -1.0
    ])
    dataset.SetProjection(WSG_84_PROJECTION)
    band = dataset.GetRasterBand(1)
    for index, val in enumerate(data):
        band.WriteArray(numpy.array([list(val)]), 0, index)
    return dataset

def make_vectors_with_id(identifier: int, areas: Set[Area], filename: str) -> None:
    poly = ogr.Geometry(ogr.wkbPolygon)

    for area in areas:
        geometry = ogr.Geometry(ogr.wkbLinearRing)
        geometry.AddPoint(area.left, area.top)
        geometry.AddPoint(area.right, area.top)
        geometry.AddPoint(area.right, area.bottom)
        geometry.AddPoint(area.left, area.bottom)
        geometry.AddPoint(area.left, area.top)
        poly.AddGeometry(geometry)

    package = ogr.GetDriverByName("GPKG").CreateDataSource(filename)
    layer = package.CreateLayer("onlylayer", geom_type=ogr.wkbPolygon)
    id_field = ogr.FieldDefn("id_no", ogr.OFTInteger)
    layer.CreateField(id_field)

    feature_definition = layer.GetLayerDefn()
    feature = ogr.Feature(feature_definition)
    feature.SetGeometry(poly)
    feature.SetField("id_no", identifier)
    layer.CreateFeature(feature)

    package.Destroy()
