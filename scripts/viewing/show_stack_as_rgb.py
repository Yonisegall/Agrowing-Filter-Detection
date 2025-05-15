from pathlib import Path

import cv2
import numpy as np

from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.show_np_image import show_image

project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing

# === Config ===
# npz_file_path = project_root / r'stacks\im020.npz'
# npz_file_path = project_root / r'stacks\before_changes1.npz'
# npz_file_path = project_root / r'stacks\im1.npz'

# npz_file_path = project_root / r'stacks\Colors.npz'
npz_file_path = project_root / r'stacks\Colors_unnormalize.npz'



# === Load data
stack = np.load(npz_file_path)['stack']
band_names = np.load(npz_file_path)['band_names']


rgb = hyperspectral_to_rgb(stack, band_names)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
show_image(bgr, "Image")

while True:
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break