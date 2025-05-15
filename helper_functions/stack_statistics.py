import numpy as np
from matplotlib import pyplot as plt


def print_stack_statistics(stack):
    # Reshape to (num_pixels, num_bands) for easier processing
    reshaped = stack.reshape(-1, stack.shape[2])  # shape: (3188*3189, 14)
    band_min = np.min(reshaped, axis=0)
    band_max = np.max(reshaped, axis=0)
    band_mean = np.mean(reshaped, axis=0)
    band_median = np.median(reshaped, axis=0)
    for i in range(stack.shape[2]):
        print(f"Band {i}: Min={band_min[i]}, Max={band_max[i]}, Mean={band_mean[i]:.2f}, Median={band_median[i]}")


def histogram_stack_statistics(stack):
    num_bands = stack.shape[2]
    reshaped = stack.reshape(-1, num_bands)  # shape: (3188*3189, 14)

    # Plot histograms
    fig, axs = plt.subplots(4, 4, figsize=(15, 10))
    axs = axs.ravel()

    for i in range(num_bands):
        axs[i].hist(reshaped[:, i], bins=100, color='gray', edgecolor='black')
        axs[i].set_title(f'Band {i}')
        axs[i].set_xlabel('Pixel value')
        axs[i].set_ylabel('Frequency')

    # Hide extra subplot if num_bands < 16
    for j in range(num_bands, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


def print_grayscale_statistics(gray_image):
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)
    mean_val = np.mean(gray_image)
    median_val = np.median(gray_image)
    print(f"gray image: Min={min_val}, Max={max_val}, Mean={mean_val:.2f}, Median={median_val}")

def histogram_grayscale_statistics(image):
    # Flatten the image
    flattened = image.flatten()

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(flattened, bins=256, color='skyblue', edgecolor='black')
    plt.title('Histogram of Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()