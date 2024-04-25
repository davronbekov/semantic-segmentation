from torch.utils.data import Dataset
from src.image_reader import ImageReader
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, data, in_classes, out_classes, preprocess, **kwargs):
        self.data = data

        self.in_classes = in_classes
        self.out_classes = out_classes

        self.preprocess = preprocess

        self.kwargs = kwargs

        self.image_reader = ImageReader('rasterio')

    def __len__(self):
        return len(self.data)

    def pre_process(self, image, mask):
        if self.preprocess:
            transformed = self.preprocess(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    def __getitem__(self, index):
        with self.image_reader.open(self.data[index]) as src:
            dimensions = src.shape
            original_width = dimensions[1]
            original_height = dimensions[0]

            x_random = np.random.randint(0, original_width - self.kwargs['slices']['width'])
            y_random = np.random.randint(0, original_height - self.kwargs['slices']['height'])

            x_start, x_stop = x_random, x_random + self.kwargs['slices']['width']
            y_start, y_stop = y_random, y_random + self.kwargs['slices']['height']

            image = src.read(self.in_classes, window=((y_start, y_stop), (x_start, x_stop)), boundless=True)
            mask = src.read(self.out_classes, window=((y_start, y_stop), (x_start, x_stop)), boundless=True)

            image = np.stack(image, axis=-1)

        # transform
        image, mask = self.pre_process(image, mask)
        mask = np.expand_dims(mask, axis=0)

        return image, mask
