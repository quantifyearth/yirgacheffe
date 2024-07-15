import os
import tempfile

import numpy as np
from osgeo import gdal

from helpers import gdal_dataset_with_data
from yirgacheffe.layers import RasterLayer
from yirgacheffe.window import Area, PixelScale

def test_simple_two_band_image() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        target_path = os.path.join(tempdir, "target.tif")

        bands = 4
        target = RasterLayer.empty_raster_layer(
            Area(-1, 1, 1, -1),
            PixelScale(1.0, 1.0),
            gdal.GDT_Byte,
            filename=target_path,
            bands=bands
        )

        for i in range(bands):
            data1 = np.full((2, 2), i+1)
            layer1 = RasterLayer(gdal_dataset_with_data((-1.0, 1.0), 1.0, data1))
            layer1.save(target, band=i+1)

        target.close()

        for i in range(bands):
            o = RasterLayer.layer_from_file(target_path, band=i+1)
            assert o.sum() == (4 * (i + 1))
