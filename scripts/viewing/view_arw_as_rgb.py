from pathlib import Path
import cv2
import numpy as np
from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.show_np_image import show_image
from helper_functions.pipeline import process_rgb_to_stack
from helper_functions.rawutils import read_arw_as_rgb
from band_values import Sextuple218mm_bands

# === Base project path
project_root = Path(__file__).resolve().parents[2]  # goes up two levels from scripts

# === Config ===
# file_path = project_root / r'data\od_TDP00020.ARW'
# file_path = project_root / r'data\M24_23_03_25_TDP00872.ARW'
# file_path = project_root / r'data\TDP00193.ARW'
file_path = project_root / r'data\yarkonim2_DSC00008.ARW'

band_names = Sextuple218mm_bands

# === Load and process ===
print("[INFO] Reading RAW image...")
rgb_image = read_arw_as_rgb(file_path)

print("[INFO] Processing RGB to spectral stack...")
stack, band_names = process_rgb_to_stack(rgb_image, cols=3, rows=2, band_names=band_names)

print("[INFO] Final stack shape:", stack.shape)
if len(band_names) == 14 and np.all(band_names == np.array(['405', '430', '450', '490', '525', '550', '560', '570', '630', '650', '685', '710', '735', '850'])):
    rgb = hyperspectral_to_rgb(stack, band_names)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    show_image(bgr, "Image")

    while True:
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
