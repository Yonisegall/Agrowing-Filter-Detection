import rawpy
import cv2
from pathlib import Path

from helper_functions.show_image_with_pixel_info import show_image_with_pixel_info

# === Path to your ARW image ===
project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing/

# === Config ===
# file_path = project_root / r"data/M24_23_03_25_TDP00872.ARW
file_path = project_root / r'data\TDP00911.ARW'
# file_path = project_root / r"data/od_TDP00020.png"


file_suffix = file_path.suffix.lower()



# === Process based on file type ===
if file_suffix == '.arw':
    # RAW file (.ARW) processing
    with rawpy.imread(str(file_path)) as raw:
        bgr = raw.postprocess()
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

elif file_suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
    # Standard image file processing
    image = cv2.imread(str(file_path))
    if image is None:
        raise Exception(f"Error: Unable to load image from {file_path}")
else:
    raise Exception(f"Unsupported file type: {file_suffix}")

# Resize to fit screen (e.g., 1280×720)
resized = cv2.resize(image, (1280, 720))

# cv2.imshow('Resized Image', resized)
show_image_with_pixel_info(resized)

cv2.waitKey(0)
cv2.destroyAllWindows()