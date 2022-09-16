from math import ceil
from typing import Set, Optional

import numpy
from osgeo import gdal, ogr

from yirgacheffe.layers import Layer, Area

def make_dataset_of_region(area: Area, pixel_pitch: float, filename: Optional[str]=None) -> gdal.Dataset:
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
	dataset.SetProjection("WGS 84")
	# the dataset isn't valid until you populate the data
	band = dataset.GetRasterBand(1)
	for yoffset in range(dataset.RasterYSize):
		band.WriteArray(numpy.array([[(yoffset % 256),] * dataset.RasterXSize]), 0, yoffset)
	return dataset

def make_vectors_with_id(id: int, areas: Set[Area], filename: str) -> None:
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
	feature.SetField("id_no", id)
	layer.CreateFeature(feature)

	package.Destroy()
