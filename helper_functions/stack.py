import numpy as np


def build_multilayer_stack(aligned_tiles):
    """
    Concatenates 6 aligned RGB tiles into a single multi-channel image.
    
    Output shape: (H, W, 18)  # 6 tiles * 3 channels
    """
    # Check they are all same size
    h, w, c = aligned_tiles[0].shape
    assert c == 3, "Expected 3 channels per tile"
    assert all(tile.shape == (h, w, 3) for tile in aligned_tiles), "All tiles must be the same size"

    # Stack along the channel axis
    return np.concatenate(aligned_tiles, axis=2)  # (H, W, 18)


def save_multilayer_npz(stack, band_names, out_path):
    """
    Saves a multi-band spectral stack (H, W, 18) to a compressed .npz file.

    Parameters:
        stack (np.ndarray): Shape (H, W, 18) — 6 RGB tiles concatenated on channel axis.
        band_names (list of str): List of 18 names (e.g. ['Band1_R', 'Band1_G', 'Band1_B', ..., 'Band6_B']).
        out_path (str): Path to output .npz file.
    """
    assert stack.ndim == 3 and stack.shape[2] == len(band_names), \
        f"Expected {stack.shape[2]} band names, got {len(band_names)}"

    np.savez_compressed(out_path, stack=stack, band_names=np.array(band_names))
    print(f"[INFO] Saved stack with {len(band_names)} bands to: {out_path}")


def remove_empty_bands_and_sort(stack, band_names):
    """
    Removes bands with empty names and sorts the remaining bands and stack layers
    by the numeric value of the band name.

    Parameters:
    - stack (np.ndarray): 3D array with shape (H, W, C), where C = number of bands.
    - band_names (List[str]): List of band names as strings, possibly with empty strings.

    Returns:
    - stack_sorted (np.ndarray): Stack with valid and sorted bands.
    - band_names_sorted (List[str]): Sorted list of valid band names.
    """
    assert stack.ndim == 3 and stack.shape[2] == len(band_names)
    valid_indices = [(i, name) for i, name in enumerate(band_names) if name.strip() != '']
    valid_indices_sorted = sorted(valid_indices, key=lambda x: float(x[1]))
    sorted_indices = [i for i, _ in valid_indices_sorted]
    sorted_band_names = [name for _, name in valid_indices_sorted]
    stack_sorted = stack[:, :, sorted_indices]
    return stack_sorted, sorted_band_names

def remove_empty_bands_and_sort_bands(band_names):
    """
    Removes bands with empty names and sorts the remaining bands
    by the numeric value of the band name.

    Parameters:
    - band_names (List[str]): List of band names as strings, possibly with empty strings.

    Returns:
    - band_names_sorted (List[str]): Sorted list of valid band names.
    """
    valid_indices = [(i, name) for i, name in enumerate(band_names) if name.strip() != '']
    valid_indices_sorted = sorted(valid_indices, key=lambda x: float(x[1]))
    sorted_band_names = [name for _, name in valid_indices_sorted]
    return sorted_band_names

def normalize_stack_0_to_255(stack):
    # """
    # Normalize each band (last dimension) of a 3D image stack to the range 0-255.
    #
    # Parameters:
    # - stack (np.ndarray): Input array of shape (H, W, B) where B is number of bands.
    #
    # Returns:
    # - np.ndarray: Normalized stack as.
    # """
    # assert isinstance(stack, np.ndarray), "'stack' must be a NumPy ndarray"
    # assert np.issubdtype(stack.dtype, np.floating), f"'stack' must be a float array, but got {stack.dtype}"
    #
    # normalized_stack = np.zeros_like(stack)
    #
    # for band in range(stack.shape[2]):
    #     band_data = stack[:, :, band]
    #
    #     min_val = band_data.min()
    #     max_val = band_data.max()
    #
    #     # Avoid division by zero if the band is flat
    #     if min_val == max_val:
    #         normalized_stack[:, :, band] = 0
    #         raise Exception
    #     else:
    #         scaled = 255 * (band_data - min_val) / (max_val - min_val)
    #         normalized_stack[:, :, band] = scaled
    #
    # return normalized_stack


    """
    Normalize the entire stack globally to the range 0-255.

    Parameters:
    - stack (np.ndarray): Input array of shape (H, W, B).

    Returns:
    - np.ndarray: Globally normalized stack.
    """
    min_val = stack.min()
    max_val = stack.max()

    if min_val == max_val:
        raise ValueError("All values in the stack are identical. Cannot normalize.")

    scaled = 255 * (stack - min_val) / (max_val - min_val)
    return scaled.astype(np.uint8)


def load_multilayer_npz(path):
    """
    Loads a .npz file containing a stacked image and its band names.

    Parameters:
        path (str): Path to the .npz file

    Returns:
        tuple:
            - stack (np.ndarray): The multi-band image (H, W, N)
            - band_names (list of str): The list of band names
    """
    data = np.load(path)
    stack = data["stack"]
    band_names = data["band_names"].tolist()

    return stack, band_names

