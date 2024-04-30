from PIL import Image


class PilReader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def open(self, src: str):
        return Image.open(src, **self.kwargs)
