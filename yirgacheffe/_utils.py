
from pathlib import Path

from osgeo import gdal

class VsimemFile:
    def __init__(self, path):
        self.path = Path("/vsimem") / path

    def __enter__(self):
        return self.path

    def __exit__(self, *args):
        try:
            gdal.Unlink(self.path)
        except RuntimeError:
            pass
