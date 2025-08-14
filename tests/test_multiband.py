import os
import tempfile

import numpy as np
from osgeo import gdal

from tests.helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer
from yirgacheffe.window import Area, PixelScale

def test_simple_two_band_image() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        target_path = os.path.join(tempdir, "target.tif")

        # Create an output raster layer with a number of bands
        bands = 4
        target = RasterLayer.empty_raster_layer(
            Area(-1, 1, 1, -1),
            PixelScale(1.0, -1.0),
            gdal.GDT_Byte,
            filename=target_path,
            bands=bands
        )

        # Create a set of rasters in turn to fill each band
        for i in range(bands):
            data1 = np.full((2, 2), i+1)
            layer1 = RasterLayer(gdal_dataset_with_data((-1.0, 1.0), 1.0, data1))
            layer1.save(target, band=i+1)

        # force things to disk
        target.close()

        #check they do what we expect
        for i in range(bands):
            o = RasterLayer.layer_from_file(target_path, band=i+1)
            assert o.sum() == (4 * (i + 1))

def test_stack_tifs_with_area_match() -> None:
    with tempfile.TemporaryDirectory() as tempdir:

        # create a set of tifs that have some intersection (note the
        # slight alignment offset when we create them)
        bands = 4
        source_layers = []
        for i in range(bands):
            data1 = np.full((100, 100), i+1)
            layer1 = RasterLayer(gdal_dataset_with_data((-100+i, 100+i), 1, data1))
            source_layers.append(layer1)

        intersection = RasterLayer.find_intersection(source_layers)
        for layer in source_layers:
            layer.set_window_for_intersection(intersection)

        layer = source_layers[-1]
        assert layer.window.xsize == 100 - (bands - 1)
        assert layer.window.ysize == 100 - (bands - 1)

        target_path = os.path.join(tempdir, "target.tif")
        target = RasterLayer.empty_raster_layer_like(layer, filename=target_path, bands=bands)
        for i in range(bands):
            source_layers[i].save(target, band=i+1)

        # force things to disk
        target.close()

        #check they do what we expect
        for i in range(bands):
            o = RasterLayer.layer_from_file(target_path, band=i+1)
            assert o.sum() == ((100 - (bands - 1)) * (100 - (bands - 1)) * (i + 1))
