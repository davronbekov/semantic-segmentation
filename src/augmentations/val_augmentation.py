import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def val_augmentations():
    return A.Compose([
        A.Normalize(mean=(106.9, 108.9, 98.4), std=(54.6, 51.0, 50.5), max_pixel_value=1, always_apply=True),
        ToTensorV2()
    ])
