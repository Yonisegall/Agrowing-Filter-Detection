from pathlib import Path

import cv2
import numpy as np

from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.show_np_image import show_image

project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing

# === Config ===

# npz_file_path = project_root / r'stacks\Colors_Un_normalize.npz'
# npz_file_path = project_root / r'stacks\Colors_normalize1.npz'
# npz_file_path = project_root / r'stacks\Colors_normalize2.npz'
# npz_file_path = project_root / r'stacks\Colors_normalize3.npz'
npz_file_path = project_root / r'stacks\Colors_normalize4.npz'


# === Load data

stack = np.load(npz_file_path)['stack']
band_names = np.load(npz_file_path)['band_names']

# stack shape: (3188, 3189, 14)
# band names: ['405' '430' '450' '490' '525' '550' '560' '570' '630' '650' '685' '710' '735' '850']

print(f'stack shape: {stack.shape}')
print(f'band names: {band_names}')

if len(band_names) == 14 and np.all(band_names == np.array(['405', '430', '450', '490', '525', '550', '560', '570', '630', '650', '685', '710', '735', '850'])):

    RGB_image = hyperspectral_to_rgb(stack, band_names)

    print(f'RGB.shape={RGB_image.shape}')

    bgr = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR)
    show_image(bgr, "Image")

    print("Maximum value in stack:", np.max(stack))

    # cv2.imshow("Image", bgr)
    while True:
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

