import os
import cv2
import numpy as np
from scipy.ndimage import minimum

image_dir = 'D:\low\**'
result_dir = 'D:\Results\low\**'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
image_len = len(image_files)
win_size = 5
r = 3
eps = 1e-7
t0 = 0.2

def get_dark_channel(image, win_size):
    m, n, _ = image.shape
    pad_size = win_size // 2

    pad_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant',
                       constant_values=np.inf)

    dark_channel = np.zeros((m, n), dtype=image.dtype)

    for j in range(m):
        for i in range(n):
            patch = pad_image[j: j + (win_size-1), i: i + (win_size-1), :]
            dark_channel[j, i] = np.min(patch)

    return dark_channel

def get_atmosphere(image, dark_channel):
    m, n, _ = image.shape
    n_pixels = m * n
    n_search_pixels = int(n_pixels * 0.01)

    dark_vec = dark_channel.reshape(n_pixels)
    image_vec = image.reshape(n_pixels, 3)

    indices = np.argsort(dark_vec)[::-1]

    accumulator = np.zeros(3)

    for k in range(n_search_pixels):
        accumulator += image_vec[indices[k]]

    atmosphere = accumulator / n_search_pixels

    return atmosphere

for i in range(image_len):
    img_name = os.path.join(image_dir, image_files[i])
    img = cv2.imread(img_name).astype(np.float64) / 255.0
    dark_channel = get_dark_channel(img, win_size)
    atmosphere = get_atmosphere(img, dark_channel)
    atmosphere = atmosphere.reshape((1, 1, 3))
    I3 = 1 - img / atmosphere
    # For the normal Image, don't need /atmosphere
    # I3 = 1 - img
    S = np.min(I3, axis=2)
    attention_name = os.path.join(result_dir, image_files[i].split('.')[0] + '_attention.png')
    cv2.imwrite(attention_name, S * 255)