from src.readers.rasterio_reader import RasterioReader


class ImageReader:
    def __init__(self, adapter: str, **kwargs):
        self.adapter = None
        if adapter == 'rasterio':
            self.adapter = RasterioReader(**kwargs)

        if not self.adapter:
            raise Exception(f'{adapter} adapter at ImageReader is not found!')

    def open(self, src: str):
        return self.adapter.open(src)
