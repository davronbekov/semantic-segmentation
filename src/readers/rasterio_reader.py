import rasterio


class RasterioReader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def open(self, src: str):
        return rasterio.open(src, **self.kwargs)
