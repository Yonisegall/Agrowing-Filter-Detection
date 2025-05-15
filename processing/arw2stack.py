from pathlib import Path
from helper_functions.pipeline import process_rgb_to_stack
from helper_functions.rawutils import read_arw_as_rgb
from helper_functions.stack import save_multilayer_npz
from band_values import Sextuple218mm_bands
from scripts.viewing.show_arw_image import show_arw

# === Base project path
project_root = Path(__file__).resolve().parents[2]  # goes up two levels from scripts

# === Config ===
band_names = Sextuple218mm_bands
# file_path = project_root / r'data\od_TDP00020.ARW'
# out_path = project_root / r'stacks\im020.npz'

# file_path = project_root / r'data\TDP00203.ARW'
# out_path = project_root / r'stacks\before_change_det11.npz'
#
# file_path = project_root / r'data\M24_23_03_25_TDP00872.ARW'
# out_path = project_root / r'stacks\im1.npz'

# file_path = project_root / r'data\yarkonim2_DSC00008.ARW'
# out_path = project_root / r'stacks\before_changes1.npz'

# file_path = project_root / r'data\DSC00029.ARW'
# out_path = project_root / r'stacks\Colors.npz'

file_path = project_root / r'data\DSC00029.ARW'
out_path = project_root / r'stacks\Colors_normalize.npz'

# file_path = project_root / r'data\yarkonim2_DSC00008.ARW'
# out_path = project_root / r'stacks\before_changes1_unnormalize.npz'

# file_path = project_root / r'data\yarkonim2_DSC00039.ARW'
# out_path = project_root / r'stacks\after_changes1.npz'

# todo: check file exists and is correct type (and has type)

# === Load and process ===
print("[INFO] Reading RAW image...")
rgb_image = read_arw_as_rgb(file_path)

print("Shape:", rgb_image.shape)
print("Dtype:", rgb_image.dtype)
print("Min value:", rgb_image.min())
print("Max value:", rgb_image.max())
print("Sample pixels (first 5x5 block):\n", rgb_image[0:5, 0:5, :])

print("[INFO] Processing RGB to spectral stack...")
stack, band_names = process_rgb_to_stack(rgb_image, cols=3, rows=2, band_names=band_names)

print("[INFO] Final stack shape:", stack.shape)

print("[INFO] Saving stack to .npz file...")
save_multilayer_npz(stack, band_names, out_path)

print("[DONE] Stack saved at:", out_path)
