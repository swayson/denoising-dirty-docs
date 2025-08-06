import pandas as pd
from sklearn.metrics import mean_squared_error
from skimage import img_as_float, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os

def reshape_as_img(x, pixel_dims):
    return x.reshape(pixel_dims)

def reshape_features(x, n_obs):
    return x.reshape((n_obs,))
    
def imshow(img, cmap=plt.cm.gray, title=None, figsize=(9,6), **kwargs):
    'helper plotting function'
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    ax.imshow(img, cmap, **kwargs)
    return

def hist(img, title='Histogram', figsize=(9,6)):
    hist = np.histogram(img_as_ubyte(img), bins=np.arange(0, 256))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)  
    if title is not None:
        ax.set_title(title)
    ax.plot(hist[1][:-1], hist[0], lw=2)
    return

def evaluate_mean_squared_error(y_true, y_pred):
    n_obs = y_true.size
    return mean_squared_error(
        reshape_features(y_true, n_obs), 
        reshape_features(y_pred, n_obs))

def to_submission_format(img, img_name):
    img_id = os.path.splitext(img_name)[0]
    rows, cols = img.shape
    submission_data = []
    for r in range(rows):
        for c in range(cols):
            submission_data.append([f"{img_id}_{r+1}_{c+1}", img[r, c]])
    return pd.DataFrame(submission_data, columns=['id', 'value'])
