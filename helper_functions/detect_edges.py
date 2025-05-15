import numpy as np
from scipy.ndimage import sobel


def detect_edges(image):
    """Detect edges using Sobel filter and normalize result."""
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    magnitude = np.hypot(dx, dy)
    return (magnitude / np.max(magnitude)) if np.max(magnitude) > 0 else magnitude