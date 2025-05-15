from pathlib import Path
import cv2
import numpy as np

from helper_functions.cube2grayscale import cube2grayscale
from helper_functions.find_homography import find_homography
from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.show_np_image import show_image, show_image_with_zoom
from helper_functions.difference_threshold_mask import calculate_changed_mask
from helper_functions.stack import save_multilayer_npz

# === Base project path
project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing/

# === Config ===
# npz_file_path1 = project_root / r'stacks\im1_norm.npz'
# npz_file_path2 = project_root / r'stacks\im2_norm.npz'

npz_file_path1 = project_root / r'stacks\before_changes1.npz'
npz_file_path2 = project_root / r'stacks\after_changes1.npz'

# === Load data
stack1 = np.load(npz_file_path1)['stack']
stack2 = np.load(npz_file_path2)['stack']
band_names = np.load(npz_file_path1)['band_names']
band_names2 = np.load(npz_file_path2)['band_names']
assert np.all(band_names == band_names2) , "band_names should be the same"

# turn to grayscale according to some logic
grayscale1 = cube2grayscale(stack1)
grayscale2 = cube2grayscale(stack2)

# compare with sift, find homography. input: two picture. output: homography
homography, homography_mask = find_homography(grayscale1, grayscale2)
h, w = grayscale1.shape
stack2_warped = cv2.warpPerspective(stack2, homography, (w, h))

show_image(cv2.cvtColor(hyperspectral_to_rgb(stack2_warped, band_names), cv2.COLOR_RGB2BGR), 'stack2_warped')
show_image(cv2.cvtColor(hyperspectral_to_rgb(stack2, band_names), cv2.COLOR_RGB2BGR), 'stack2')
show_image(cv2.cvtColor(hyperspectral_to_rgb(stack1, band_names), cv2.COLOR_RGB2BGR), 'stack1')

# Create a white mask the same size as the grayscale input
valid_mask = np.ones_like(grayscale2, dtype=np.uint8) * 255  # shape: (H, W), values: 255
# Warp the valid region mask using the same homography
warped_valid_mask = cv2.warpPerspective(valid_mask, homography, (w, h))  # shape: (H, W), values: 255 where valid
# Create a boolean mask: True where the warped pixel is invalid (i.e. warped from nothing)
invalid_mask = warped_valid_mask == 0  # shape: (H, W), True where invalid

# stack1[invalid_mask] = [0] * len(band_names)
# stack2_warped[invalid_mask] = [0] * len(band_names)

changed_mask = calculate_changed_mask(stack1, stack2_warped)
# changed_rgb = np.stack([cube2grayscale(stack1)] * 3, axis=-1).astype(np.uint8) # int rounds from 0 to 1
# changed_rgb = np.stack(hyperspectral_to_rgb(stack1), axis=-1).astype(np.uint8) # int rounds from 0 to 1
changed_rgb = cv2.cvtColor(hyperspectral_to_rgb(stack1, band_names), cv2.COLOR_RGB2BGR)
changed_rgb[changed_mask] = [0, 0, 255]
changed_rgb[invalid_mask] = [0, 0, 0]

show_image_with_zoom(changed_rgb, 'differences')
# save_multilayer_npz(stack2_warped, band_names, project_root / r'stacks\stack2_norm_warped.npz')

cv2.waitKey(0)
cv2.destroyAllWindows()
