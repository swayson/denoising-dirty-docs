# Denoising Dirty Documents

This project is an attempt to denoise dirty documents using a deep learning approach. It was originally created for a [Kaggle competition](https://www.kaggle.com/c/denoising-dirty-documents).

This updated version refactors the original exploratory notebook into a more structured Python project and implements a convolutional autoencoder for denoising.

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Contains the original exploratory Jupyter notebook.
- `src/`: Contains the Python source code for the denoising pipeline.
  - `data_loader.py`: Loads and preprocesses the data.
  - `model.py`: Defines the convolutional autoencoder model.
  - `train.py`: Trains the denoising model.
  - `evaluate.py`: Evaluates the model and saves sample denoised images.
  - `helpers.py`: Contains helper functions for plotting and evaluation.
- `model/`: Stores the trained model file.
- `output/`: Stores the output comparison images.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model:**
   ```bash
   python src/train.py
   ```
   This will train the convolutional autoencoder and save the trained model to `model/denoising_autoencoder.h5`.

2. **Evaluate the model:**
   ```bash
   python src/evaluate.py
   ```
   This will load the trained model, denoise the test images, print the Mean Squared Error, and save comparison images in the `output/` directory.
