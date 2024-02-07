from typing import Set, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr

from yirgacheffe.window import Area
from yirgacheffe.layers import YirgacheffeLayer
from yirgacheffe.rounding import round_up_pixels
from yirgacheffe import WGS_84_PROJECTION

def gdal_dataset_of_region(area: Area, pixel_pitch: float, filename: Optional[str]=None) -> gdal.Dataset:
    if filename:
        driver = gdal.GetDriverByName('GTiff')
    else:
        driver = gdal.GetDriverByName('mem')
        filename = 'mem'
    dataset = driver.Create(
        filename,
        round_up_pixels(((area.right - area.left) / pixel_pitch), pixel_pitch),
        round_up_pixels(((area.top - area.bottom) / pixel_pitch), pixel_pitch),
        1,
        gdal.GDT_Byte,
        []
    )
    dataset.SetGeoTransform([
        area.left, pixel_pitch, 0.0, area.top, 0.0, pixel_pitch * -1.0
    ])
    dataset.SetProjection(WGS_84_PROJECTION)
    # the dataset isn't valid until you populate the data
    band = dataset.GetRasterBand(1)
    for yoffset in range(dataset.RasterYSize):
        band.WriteArray(np.array([[(yoffset % 256),] * dataset.RasterXSize]), 0, yoffset)
    return dataset

def gdal_empty_dataset_of_region(area: Area, pixel_pitch: float) -> gdal.Dataset:
    dataset = gdal.GetDriverByName('mem').Create(
        'mem',
        round_up_pixels(((area.right - area.left) / pixel_pitch), pixel_pitch),
        round_up_pixels(((area.top - area.bottom) / pixel_pitch), pixel_pitch),
        0,
        gdal.GDT_Byte,
        []
    )
    dataset.SetGeoTransform([
        area.left, pixel_pitch, 0.0, area.top, 0.0, pixel_pitch * -1.0
    ])
    dataset.SetProjection(WGS_84_PROJECTION)
    return dataset

def gdal_dataset_of_layer(layer: YirgacheffeLayer, filename: Optional[str]=None) -> gdal.Dataset:
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

def gdal_dataset_with_data(origin: Tuple, pixel_pitch: float, data: np.array) -> gdal.Dataset:
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
    dataset.SetProjection(WGS_84_PROJECTION)
    band = dataset.GetRasterBand(1)
    for index, val in enumerate(data):
        band.WriteArray(np.array([list(val)]), 0, index)
    return dataset

def gdal_multiband_dataset_with_data(origin: Tuple, pixel_pitch: float, datas: list[np.array]) -> gdal.Dataset:
    for data in datas:
        assert data.ndim == 2
    datatype = gdal.GDT_Byte
    if isinstance(data[0][0], float):
        datatype = gdal.GDT_Float64
    dataset = gdal.GetDriverByName('mem').Create(
        'mem',
        len(datas[0][0]),
        len(datas[0]),
        len(datas),
        datatype,
        []
    )
    dataset.SetGeoTransform([
        origin[0], pixel_pitch, 0.0, origin[1], 0.0, pixel_pitch * -1.0
    ])
    dataset.SetProjection(WGS_84_PROJECTION)
    for i in range(len(datas)):
        band = dataset.GetRasterBand(i + 1)
        for index, val in enumerate(datas[i]):
            band.WriteArray(np.array([list(val)]), 0, index)
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

    srs = ogr.osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    package = ogr.GetDriverByName("GPKG").CreateDataSource(filename)
    layer = package.CreateLayer("onlylayer", srs, geom_type=ogr.wkbPolygon)
    id_field = ogr.FieldDefn("id_no", ogr.OFTInteger)
    layer.CreateField(id_field)

    feature_definition = layer.GetLayerDefn()
    feature = ogr.Feature(feature_definition)
    feature.SetGeometry(poly)
    feature.SetField("id_no", identifier)
    layer.CreateFeature(feature)

    package.Destroy()

def make_vectors_with_mutlile_ids(areas: Set[Tuple[Area,int]], filename: str) -> None:
    srs = ogr.osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    package = ogr.GetDriverByName("GPKG").CreateDataSource(filename)
    layer = package.CreateLayer("onlylayer", srs, geom_type=ogr.wkbPolygon)
    field_type = ogr.OFTInteger
    try:
        if isinstance(list(areas)[0][1], float):
            field_type = ogr.OFTReal
    except IndexError:
        pass
    id_field = ogr.FieldDefn("id_no", field_type)
    layer.CreateField(id_field)
    feature_definition = layer.GetLayerDefn()

    for area, identifier in areas:
        poly = ogr.Geometry(ogr.wkbPolygon)
        geometry = ogr.Geometry(ogr.wkbLinearRing)
        geometry.AddPoint(area.left, area.top)
        geometry.AddPoint(area.right, area.top)
        geometry.AddPoint(area.right, area.bottom)
        geometry.AddPoint(area.left, area.bottom)
        geometry.AddPoint(area.left, area.top)
        poly.AddGeometry(geometry)

        feature = ogr.Feature(feature_definition)
        feature.SetGeometry(poly)
        feature.SetField("id_no", identifier)
        layer.CreateFeature(feature)

    package.Destroy()

def generate_child_tile(xoffset: int, yoffset: int, width: int, height: int, outer_width: int, outer_height: int) -> np.ndarray:
    data = np.zeros((height, width))
    assert xoffset + width <= outer_width
    assert yoffset + height <= outer_height
    for y in range(height):
        for x in range(width):
            data[y][x] = ((yoffset + y) * outer_width) + (x + xoffset)
    return data

def make_vectors_with_empty_feature(areas: Set[Tuple[Area,int]], filename: str) -> None:
    srs = ogr.osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    package = ogr.GetDriverByName("GPKG").CreateDataSource(filename)
    layer = package.CreateLayer("onlylayer", srs, geom_type=ogr.wkbPolygon)
    field_type = ogr.OFTInteger
    try:
        if isinstance(list(areas)[0][1], float):
            field_type = ogr.OFTReal
    except IndexError:
        pass
    id_field = ogr.FieldDefn("id_no", field_type)
    layer.CreateField(id_field)
    feature_definition = layer.GetLayerDefn()

    for area, identifier in areas:
        poly = ogr.Geometry(ogr.wkbPolygon)
        geometry = ogr.Geometry(ogr.wkbLinearRing)
        geometry.AddPoint(area.left, area.top)
        geometry.AddPoint(area.right, area.top)
        geometry.AddPoint(area.right, area.bottom)
        geometry.AddPoint(area.left, area.bottom)
        geometry.AddPoint(area.left, area.top)
        poly.AddGeometry(geometry)

        feature = ogr.Feature(feature_definition)
        feature.SetGeometry(poly)
        feature.SetField("id_no", identifier)
        layer.CreateFeature(feature)

    feature = ogr.Feature(feature_definition)
    layer.CreateFeature(feature)

    package.Destroy()