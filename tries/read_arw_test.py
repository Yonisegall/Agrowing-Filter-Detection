import numpy as np
import rawpy
import cv2
from pathlib import Path

from helper_functions.show_image_with_pixel_info import show_image_with_pixel_info
from helper_functions.stack import normalize_stack_0_to_255
from helper_functions.stack_statistics import print_stack_statistics, histogram_stack_statistics

# === Path to your ARW image ===
project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing/

# === Config ===
# file_path = project_root / r"data/M24_23_03_25_TDP00872.ARW
file_path = project_root / r'data\yarkonim2_DSC00008.ARW'
# file_path = project_root / r"data/od_TDP00020.png"
file_suffix = file_path.suffix.lower()

# === Process based on file type ===
if file_suffix == '.arw':
    pass
else:
    raise Exception(f"Unsupported file type: {file_suffix}")

# RAW file (.ARW) processing
with rawpy.imread(str(file_path)) as raw:
    bgr = raw.postprocess()
bgr = bgr.astype(np.float32)

image1 = normalize_stack_0_to_255(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
print_stack_statistics(image1)
histogram_stack_statistics(image1)
# time.sleep(99999999)
# # print(f'type(band_data)={type(image1)}{image1.dtype}')
# print(f'image1:')
# print_stack_statistics(image1)

# with rawpy.imread(str(file_path)) as raw:
#     rgb = raw.postprocess(
#         use_camera_wb=False,
#         no_auto_bright=True,
#         output_bps=16,  # preserve dynamic range
#         gamma=(1, 1)  # linear gamma
#     )
#     # current pixel range is [0, 65535], change it to [0, 255]
# rgb = rgb.astype(np.float32) * (255 / 65535)
# histogram_stack_statistics(rgb)
# print_stack_statistics(rgb)
# show_image_with_pixel_info(rgb)

# image2 = normalize_stack_0_to_255(rgb)
# show_image_with_pixel_info(image2)
# histogram_stack_statistics(image2)

resized1 = cv2.resize(image1, (1280, 720))
show_image_with_pixel_info(resized1/255)

# resized2 = cv2.resize(image2, (1280, 720))
# show_image_with_pixel_info(resized2)

cv2.waitKey(0)
cv2.destroyAllWindows()