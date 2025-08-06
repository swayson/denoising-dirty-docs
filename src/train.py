import os
import numpy as np
from data_loader import load_data
from model import build_autoencoder

IMAGE_SIZE = (256, 256)

def train():
    # Load data
    print("Loading data...")
    X_train, y_train = load_data('data', image_size=IMAGE_SIZE)

    # Normalize and reshape data
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_train = np.reshape(X_train, (len(X_train), IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    y_train = np.reshape(y_train, (len(y_train), IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    # Build model
    print("Building model...")
    autoencoder = build_autoencoder(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    # Train model
    print("Training model...")
    autoencoder.fit(X_train, y_train,
                    epochs=10,
                    batch_size=16,
                    shuffle=True)

    # Save model
    print("Saving model...")
    if not os.path.exists('model'):
        os.makedirs('model')
    autoencoder.save('model/denoising_autoencoder.h5')

if __name__ == '__main__':
    train()
