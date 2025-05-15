import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
import copy

def organize_spectra_by_material(material_data):
    spectra_by_material = {}
    for entry in material_data:
        material = entry['material']
        spectra = np.array(entry['spectra'])  # Shape: (num_samples, 14)
        spectra_by_material[material] = spectra
    return spectra_by_material

def clac_mean_and_covariance(spectra_by_material):
    mean_and_covariance = {}
    for material, spectra in spectra_by_material.items():
        mu = np.mean(spectra, axis=0)  # Mean vector (14,)
        cov = np.cov(spectra, rowvar=False)  # Covariance matrix (14, 14)
        mean_and_covariance[material] = {'mu': mu, 'cov': cov}
    return mean_and_covariance

def calc_log_likelihoods(image, mean_and_covariance):
    height, width, bands = image.shape
    num_pixels = height * width
    pixels = image.reshape(-1, bands)  # Shape: (num_pixels, 14)

    materials = list(mean_and_covariance.keys())
    num_materials = len(materials)
    log_likelihoods = np.zeros((num_pixels, num_materials))

    # Compute log-likelihoods
    for idx, material in enumerate(materials):
        mu = mean_and_covariance[material]['mu']
        cov = mean_and_covariance[material]['cov']
        rv = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
        log_likelihoods[:, idx] = rv.logpdf(pixels)

    return log_likelihoods

def visualize_classification(materials, material_data, classified_image, original_image):
    # Step 1: Create a mapping from material to color
    material_colors = {entry['material']: entry['color'] for entry in material_data}
    colors = [material_colors[material] for material in materials]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(materials) + 1) - 0.5, len(materials))

    # Step 2: Create the modified original image with pixel overlays
    overlay_image = copy.deepcopy(original_image)
    for entry in material_data:
        color_rgb = to_rgb(entry['color'])  # convert color name to RGB
        for y, x in entry['pixels']:
            overlay_image[y, x] = np.array(color_rgb) * 255  # scale to 0-255 range

    # Step 3: Setup the figure and pre-render all three images
    fig, ax = plt.subplots(figsize=(10, 10))

    im_classified = ax.imshow(classified_image, cmap=cmap, norm=norm)
    im_original = ax.imshow(original_image)
    im_overlay = ax.imshow(overlay_image)

    # Only classified image is visible initially
    im_original.set_visible(False)
    im_overlay.set_visible(False)

    ax.set_axis_off()
    ax.set_title('Material Classification Map')

    # Colorbar
    cbar = fig.colorbar(im_classified, ax=ax, ticks=np.arange(len(materials)))
    cbar.ax.set_yticklabels(materials)
    cbar.set_label('Material Classification', rotation=270, labelpad=15)

    def on_key(event):
        key = event.key
        if key == 'up':  # Show classified image
            im_classified.set_visible(True)
            im_original.set_visible(False)
            im_overlay.set_visible(False)
            ax.set_title('Material Classification Map')

        elif key == 'down':  # Show original image
            im_classified.set_visible(False)
            im_original.set_visible(True)
            im_overlay.set_visible(False)
            ax.set_title('Original Image')

        elif key == 'left':  # Show overlay image
            im_classified.set_visible(False)
            im_original.set_visible(False)
            im_overlay.set_visible(True)
            ax.set_title('Original Image with Material Highlights')

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()
