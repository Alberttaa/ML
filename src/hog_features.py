from __future__ import annotations

import cv2
import numpy as np


def _descriptor_for_shape(height, width):
    cell_size = 4 if min(height, width) <= 32 else 8
    block_size = cell_size * 2
    win_size = (width, height)
    block = (block_size, block_size)
    stride = (cell_size, cell_size)
    cell = (cell_size, cell_size)
    return cv2.HOGDescriptor(win_size, block, stride, cell, 9)


def extract_hog_features(images):
    if len(images) == 0:
        return np.empty((0, 0), dtype=np.float32)

    features = []
    for image in images:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.uint8)
        descriptor = _descriptor_for_shape(image.shape[0], image.shape[1])
        hog_vector = descriptor.compute(image)
        features.append(hog_vector.flatten())

    return np.asarray(features, dtype=np.float32)
