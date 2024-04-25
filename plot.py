# from src.vizualization import plot_results
# from src.readers.rasterio_reader import read_image
# from src.augmentations import test_augmentations
#
# reader = read_image(
#     [1, 2, 3],
#     [4,
#     image='./data/rooftops_ru/test/Mytishi.tif',
#     type='val'
# )
#
# item = reader[11509, 10764]
# transformed = test_augmentations()(image=item[0], mask=item[1])
# image = transformed['image']
# mask = transformed['mask']
#
# plot_results(image, mask, mask)
