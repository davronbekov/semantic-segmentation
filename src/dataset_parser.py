import os


def scan_with_folders(dataset_path):
    scanned_images = {}

    for folder in os.listdir(dataset_path):
        scanned_images[folder] = scan_files(f'{dataset_path}/{folder}')

    return scanned_images


def scan_files(dataset_path):
    scanned_images = []

    for image in os.listdir(dataset_path):
        scanned_images.append(image)

    return scanned_images
