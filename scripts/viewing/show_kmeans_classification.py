import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.material_classification import visualize_classification
from helper_functions.material_color_options import ORIGINAL_COLOR_OPTIONS

# --- Load hyperspectral stack and band names ---
project_root = Path(__file__).resolve().parents[2]
npz_file_path = project_root / 'stacks' / 'im1.npz'
npz_file_path = project_root / 'stacks' / 'before_changes1.npz'
stack = np.load(npz_file_path)['stack']             # shape: (H, W, B)
band_names = np.load(npz_file_path)['band_names']   # shape: (B,)

# --- Reshape for K-Means: pixels as rows, bands as columns ---
H, W, B = stack.shape
reshaped_stack = stack.reshape(-1, B)  # shape: (H*W, B)
reshaped_stack = reshaped_stack / np.clip(reshaped_stack.sum(axis=1, keepdims=True), a_min=1e-6, a_max=None) # this normalizes each pixel


# --- Apply K-Means clustering ---
k = 25  # Choose based on expected material count
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
material_indices = kmeans.fit_predict(reshaped_stack)

# --- Reshape result back to 2D image ---
classified_image = material_indices.reshape(H, W)

# --- Generate dummy material data for visualization ---
# Since K-Means clusters are unlabeled, we use generic names and colors
materials = [f'Material {i}' for i in range(k)]
material_data = [
    {
        "material": f"Material {i}", "color": ORIGINAL_COLOR_OPTIONS[i%20], "pixels": []
    }
    for i in range(k)
]

# --- Convert hyperspectral image to RGB for visualization ---
rgb_image = hyperspectral_to_rgb(stack, band_names)

# --- Visualize the classification ---
visualize_classification(materials, material_data, classified_image, rgb_image)
