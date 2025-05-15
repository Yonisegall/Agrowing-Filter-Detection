import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
from helper_functions.stack import normalize_stack_0_to_255

project_root = Path(__file__).resolve().parents[1]

# def hyperspectral_to_rgb(hyp_stack):
#     # Normalize to [0, 1] if needed
#     hyp_stack = hyp_stack.astype(np.float32)
#     hyp_stack -= hyp_stack.min()
#     hyp_stack /= hyp_stack.max()
#
#     # Define indices for RGB bands
#     blue_indices = [2, 3]         # 450, 490
#     green_indices = [4, 5, 6, 7]  # 525, 550, 560, 570
#     red_indices = [8, 9, 10]      # 630, 650, 685
#
#     # Average over selected bands for each color
#     blue = hyp_stack[:, :, blue_indices].mean(axis=2)
#     green = hyp_stack[:, :, green_indices].mean(axis=2)
#     red = hyp_stack[:, :, red_indices].mean(axis=2)
#
#     # Stack into RGB image
#     rgb_image = np.stack([red, green, blue], axis=2)
#
#     # Optional: Clip and rescale to [0, 255] if you want to display or save
#     rgb_image_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
#
#     return rgb_image_uint8


def hyperspectral_to_rgb(hyperspectral_stack, band_names):

    # === Load the CIE 1931 XYZ Color Matching Functions ===

    # Load the CSV
    cmf_data = pd.read_csv(project_root / 'CIE_1931_color_space.csv',
                           comment='#',  # Skip lines starting with #
                           header=0,  # Use the first non-comment line as header
                           skipinitialspace=True)  # Handle potential spaces after commas


    # Extract
    wavelengths_cmf = cmf_data['wavelength'].values  # (380 to 780 nm)
    X_cmf = cmf_data['x_bar'].values
    Y_cmf = cmf_data['y_bar'].values
    Z_cmf = cmf_data['z_bar'].values

    # === Interpolate XYZ to your bands ===
    X_interp = interp1d(wavelengths_cmf, X_cmf, bounds_error=False, fill_value=0)
    Y_interp = interp1d(wavelengths_cmf, Y_cmf, bounds_error=False, fill_value=0)
    Z_interp = interp1d(wavelengths_cmf, Z_cmf, bounds_error=False, fill_value=0)

    X_weights = X_interp(band_names)
    Y_weights = Y_interp(band_names)
    Z_weights = Z_interp(band_names)

    # === Normalize the weights ===
    X_weights /= X_weights.sum()
    Y_weights /= Y_weights.sum()
    Z_weights /= Z_weights.sum()

    # === Build transformation matrix ===
    xyz_matrix = np.stack([X_weights, Y_weights, Z_weights], axis=0)  # Shape (3, 14)

    # === Now map your hyperspectral cube ===
    h, w, bands = hyperspectral_stack.shape
    flattened = hyperspectral_stack.reshape(-1, bands)  # (h*w, 14)
    xyz_flat = flattened @ xyz_matrix.T  # (h*w, 3)
    # xyz_image = xyz_flat.reshape(h, w, 3)

    # === Optional: Convert XYZ to RGB (using sRGB standard matrix) ===
    M_xyz_to_rgb = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    rgb_flat = xyz_flat @ M_xyz_to_rgb.T
    rgb_image = rgb_flat.reshape(h, w, 3)

    # === Clip to [0, 1] range and scale if needed ===
    rgb_image = np.clip(rgb_image, 0, 255)
    rgb_uint8 = np.uint8(rgb_image)

    return rgb_uint8

