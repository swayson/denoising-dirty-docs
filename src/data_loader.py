import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def load_data(data_dir, image_size=(256, 256)):
    train_dir = os.path.join(data_dir, 'train')
    train_cleaned_dir = os.path.join(data_dir, 'train_cleaned')

    train_images = []
    train_cleaned_images = []

    image_files = [f for f in os.listdir(train_dir) if f.endswith('.png')]

    for img_name in image_files:
        img_path = os.path.join(train_dir, img_name)
        img = imread(img_path, as_gray=True)
        train_images.append(resize(img, image_size))

        cleaned_img_path = os.path.join(train_cleaned_dir, img_name)
        cleaned_img = imread(cleaned_img_path, as_gray=True)
        train_cleaned_images.append(resize(cleaned_img, image_size))

    return np.array(train_images), np.array(train_cleaned_images)

def load_test_data(data_dir, image_size=(256, 256)):
    test_dir = os.path.join(data_dir, 'test')
    test_images = []
    test_image_names = []

    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]

    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        img = imread(img_path, as_gray=True)
        test_images.append(resize(img, image_size))
        test_image_names.append(img_name)

    return np.array(test_images), test_image_names
