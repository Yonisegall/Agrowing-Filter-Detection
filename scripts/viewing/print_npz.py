from pathlib import Path

import cv2
import numpy as np

from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.show_np_image import show_image

project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing

# === Config ===
# npz_file_path = project_root / r'stacks\im1_norm.npz'
# npz_file_path = project_root / r'stacks\before_changes1_unnormalize.npz'
# npz_file_path = project_root / r'stacks\before_changes1.npz'

# npz_file_path = project_root / r'stacks\Colors.npz'
npz_file_path = project_root / r'stacks\Colors_normalize.npz'



# === Load data
stack = np.load(npz_file_path)['stack']
band_names = np.load(npz_file_path)['band_names']
# stack shape: (3188, 3189, 14)
# band names: ['405' '430' '450' '490' '525' '550' '560' '570' '630' '650' '685' '710' '735' '850']

print(f'stack shape: {stack.shape}')
print(f'band names: {band_names}')
if len(band_names) == 14 and np.all(band_names == np.array(['405', '430', '450', '490', '525', '550', '560', '570', '630', '650', '685', '710', '735', '850'])):
    rgb = hyperspectral_to_rgb(stack, band_names)
    print(f'rgb.shape={rgb.shape}')
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    show_image(bgr, "Image")
    # cv2.imshow("Image", bgr)
    while True:
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break