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

        self.image_reader = ImageReader('pil')

    def __len__(self):
        return len(self.data)

    def pre_process(self, image, mask):
        if self.preprocess:
            transformed = self.preprocess(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    def __getitem__(self, index):
        image_file = f"{self.kwargs['dataset_path']['images']}/{self.data[index]}"
        mask_file = f"{self.kwargs['dataset_path']['masks']}/{self.data[index][:-4]}_mask.png"

        with self.image_reader.open(image_file) as src:
            image = np.array(src.convert('RGB'))

        with self.image_reader.open(mask_file) as src:
            mask = np.array(src.convert('RGB'))

        # transform
        image, mask = self.pre_process(image, mask)
        mask = mask.permute(2, 0, 1)

        return image, mask
