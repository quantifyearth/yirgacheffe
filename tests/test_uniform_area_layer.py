import os
import tempfile

from helpers import gdal_dataset_of_region
from yirgacheffe.layers import UniformAreaLayer
from yirgacheffe.window import Area, Window

# A UniformAreaLayer is a hack for the fact that we have these raster layers
# that represent the area of each pixel in WSG84, and they're all the same
# value across the layer, but the GeoTIFFs are super slow to unpack. Thus the
# work around is to make these GeoTIFFS that are just one column of pixels,
# and this layer pretends that it's a 360 degree wide layer.

def test_open_uniform_area_layer() -> None:
	with tempfile.TemporaryDirectory() as tempdir:
		path = os.path.join(tempdir, "test.tif")
		area = Area(-180, 90, -179, -90)
		dataset = gdal_dataset_of_region(area, 1.0, filename=path)
		assert dataset.RasterXSize == 1
		assert dataset.RasterYSize == 180
		del dataset

		layer = UniformAreaLayer.layer_from_file(path)
		assert layer.pixel_scale == (1.0, -1.0)
		assert layer.area == Area(-180, 90, 180, -90)
		assert layer.window == Window(0, 0, 360, 180)

def test_set_intersection() -> None:
	with tempfile.TemporaryDirectory() as tempdir:
		path = os.path.join(tempdir, "test.tif")
		area = Area(-180, 90, -179, -90)
		dataset = gdal_dataset_of_region(area, 1.0, filename=path)
		assert dataset.RasterXSize == 1
		assert dataset.RasterYSize == 180
		del dataset

		layer = UniformAreaLayer.layer_from_file(path)
		layer.set_window_for_intersection(Area(-10, 10, 10, -10))
		assert layer.area == Area(-10, 10, 10, -10)
		assert layer.window == Window(170, 80, 20, 20)

		layer.reset_window()
		assert layer.area == Area(-180, 90, 180, -90)
		assert layer.window == Window(0, 0, 360, 180)
