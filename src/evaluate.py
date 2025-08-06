import os
import numpy as np
import pandas as pd
from keras.models import load_model
from data_loader import load_test_data
from helpers import to_submission_format

IMAGE_SIZE = (256, 256)

def evaluate():
    # Load data
    print("Loading test data...")
    X_test, test_image_names = load_test_data('data', image_size=IMAGE_SIZE)

    # Normalize and reshape data
    X_test = X_test.astype('float32')
    X_test = np.reshape(X_test, (len(X_test), IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    # Load model
    print("Loading model...")
    autoencoder = load_model('model/denoising_autoencoder.h5')

    # Denoise images
    print("Denoising images...")
    denoised_images = autoencoder.predict(X_test)

    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame()
    for i, img_name in enumerate(test_image_names):
        denoised_img = denoised_images[i].reshape(IMAGE_SIZE)
        submission = pd.concat([submission, to_submission_format(denoised_img, img_name)])

    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == '__main__':
    evaluate()
