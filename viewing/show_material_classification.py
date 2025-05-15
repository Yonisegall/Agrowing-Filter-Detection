import json
import numpy as np
from pathlib import Path

from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.material_classification import organize_spectra_by_material, calc_log_likelihoods, \
    visualize_classification, clac_mean_and_covariance
from helper_functions.stack_statistics import  histogram_grayscale_statistics

project_root = Path(__file__).resolve().parents[2]

# npz_file_path = project_root / project_root / r'stacks\before_change_det.npz'
npz_file_path = project_root / project_root / r'stacks\im020.npz'
# npz_file_path = project_root / project_root / r'stacks\im1.npz'
# npz_file_path = project_root / project_root / r'stacks\before_changes1.npz'
stack = np.load(npz_file_path)['stack']
band_names = np.load(npz_file_path)['band_names']
material_tag = r'im1.json'
with open(project_root / 'material_labeling' / material_tag, 'r') as f:
    material_data = json.load(f)
# todo: make sure band names from npz_file_path and material_data are the same

spectra_by_material = organize_spectra_by_material(material_data)
mean_and_covariance = clac_mean_and_covariance(spectra_by_material)
log_likelihoods = calc_log_likelihoods(stack, mean_and_covariance)
# histogram_grayscale_statistics(log_likelihoods)
materials = list(mean_and_covariance.keys())

# --- Step 1: Set log-likelihood threshold ---
log_likelihood_threshold = -100  # Tune this based on your data

# --- Step 2: Classify each pixel by highest log-likelihood ---
max_log_likelihoods = np.max(log_likelihoods, axis=1)
material_indices = np.argmax(log_likelihoods, axis=1)

# --- Step 3: Mark low-likelihood pixels as 'unknown' with a temp label -1 ---
material_indices[max_log_likelihoods < log_likelihood_threshold] = -1

# --- Step 4: Remap labels so known materials are unchanged, unknown is last ---
# known: 0, 1, ..., N-1
# unknown: N (last index)
num_known = len(mean_and_covariance)
material_indices[material_indices == -1] = num_known  # move unknown to end
classified_image = material_indices.reshape(stack.shape[:2])

# --- Step 5: Create materials and material_data (Unknown at end) ---
materials = list(mean_and_covariance.keys()) + ['Unknown']
material_data = material_data + [{
    'material': 'Unknown',
    'color': 'black',  # gray or any neutral color
    'samples': [],
    'pixels': []
}]



visualize_classification(materials, material_data, classified_image, hyperspectral_to_rgb(stack, band_names))
