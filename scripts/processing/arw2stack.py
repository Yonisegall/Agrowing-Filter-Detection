from pathlib import Path
from helper_functions.pipeline import process_rgb_to_stack
from helper_functions.rawutils import ARW_as_Matrix
from helper_functions.stack import save_multilayer_npz
from band_values import Sextuple218mm_bands
from scripts.viewing.show_arw_image import show_arw

# === Base project path
project_root = Path(__file__).resolve().parents[2]  # goes up two levels from scripts

# === Config ===
band_names = Sextuple218mm_bands

# file_path = project_root / r'data\DSC00028.ARW'
# out_path = project_root / r'stacks\Colors_Un_normalize.npz'

# file_path = project_root / r'data\DSC00028.ARW'
# out_path = project_root / r'stacks\Colors_normalize1.npz'

# file_path = project_root / r'data\DSC00028.ARW'
# out_path = project_root / r'stacks\Colors_normalize2.npz'

# file_path = project_root / r'data\DSC00028.ARW'
# out_path = project_root / r'stacks\Colors_normalize3.npz'

file_path = project_root / r'data\DSC00028.ARW'
out_path = project_root / r'stacks\Colors_normalize4.npz'


# todo: check file exists and is correct type (and has type)

# === Load and process ===
print("[INFO] Reading RAW image...")
RGB_image = ARW_as_Matrix(file_path)

print("Shape:", RGB_image.shape)
print("Dtype:", RGB_image.dtype)
print("Min value:", RGB_image.min())
print("Max value:", RGB_image.max())
print("Sample pixels (first 10x10 block):\n", RGB_image[0:5, 0:5, :])

print("[INFO] Processing RGB to spectral stack...")
stack, band_names = process_rgb_to_stack(RGB_image, cols=3, rows=2, band_names=band_names)

print("[INFO] Final stack shape:", stack.shape)

print("[INFO] Saving stack to .npz file...")
save_multilayer_npz(stack, band_names, out_path)

print("[DONE] Stack saved at:", out_path)
