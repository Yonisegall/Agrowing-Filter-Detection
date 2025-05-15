import cv2
import numpy as np
from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from band_values import Sextuple218mm_bands

def calculate_changed_mask(img1: np.ndarray, img2: np.ndarray, method='by_avg') -> np.ndarray:
    """
    Computes a binary mask indicating the differences between two images.

    Parameters:
    - img1 (np.ndarray): First hyperspectral image of shape (H, W, 14).
    - img2 (np.ndarray): Second hyperspectral image of shape (H, W, 14).
    - threshold (float): Threshold for the differences

    Returns:
    - np.ndarray: Binary mask of shape (H, W), dtype=bool.
    """
    threshold = {
        'by_avg': 19000,
        'by_hue': 150,
        'by_sam': 0.2,
    }
    # Compute per-band difference
    if method == 'by_avg':
        diff = img1 - img2
        squared_diff = diff ** 2
        sum_squared_diff = np.sum(squared_diff, axis=-1)
        # sum_diff = np.sum(diff, axis=-1)
        mask = sum_squared_diff > threshold['by_avg']
    elif method == 'by_hue':
        rgb1 = hyperspectral_to_rgb(img1, Sextuple218mm_bands) # todo: use general bands
        rgb2 = hyperspectral_to_rgb(img2, Sextuple218mm_bands)
        h1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2HSV)[...,0]
        h2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2HSV)[...,0]
        diff = h1 - h2
        mask = diff > threshold['by_hue']
    elif method == 'by_sam':
        # Normalize both images to unit spectral vectors
        norm1 = np.linalg.norm(img1, axis=-1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(img2, axis=-1, keepdims=True) + 1e-8
        unit1 = img1 / norm1
        unit2 = img2 / norm2

        # Compute cosine similarity and then angle
        dot_product = np.sum(unit1 * unit2, axis=-1)
        cos_angle = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        mask = angle > threshold['by_sam']
    else:
        raise Exception('not a method')

    return mask
