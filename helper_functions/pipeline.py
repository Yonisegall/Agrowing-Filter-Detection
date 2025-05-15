from helper_functions.split import split_image_to_tiles
from helper_functions.align import align_to_reference
from helper_functions.stack import build_multilayer_stack, remove_empty_bands_and_sort, normalize_stack_0_to_255


def process_rgb_to_stack(rgb_image, cols, rows, band_names):
    """
    Full pipeline: Split → Align → Stack → Filter

    Parameters:
        rgb_image (np.ndarray): The input RGB image (H, W, 3)
        cols (int): Number of horizontal tiles
        rows (int): Number of vertical tiles
        band_names (list of str): Should have length = cols * rows * 3

    Returns:
        (stack_filtered, band_names_filtered): Tuple of (H, W, N) array and valid band name list
    """
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, "Expected RGB image"
    assert len(band_names) == cols * rows * 3, f"Expected {cols*rows*3} band names, got {len(band_names)}"
    print("[INFO] Splitting image...")
    tiles = split_image_to_tiles(rgb_image, cols=cols, rows=rows)

    print("[INFO] Aligning to top-left tile...")
    ref_tile = tiles[0]
    aligned_tiles = [ref_tile] + [align_to_reference(ref_tile, tile) for tile in tiles[1:]]

    print("[INFO] Building stacked image...")
    stack = build_multilayer_stack(aligned_tiles)

    print("[INFO] Filtering empty band names...")
    stack_filtered, band_names_filtered = remove_empty_bands_and_sort(stack, band_names)

    print("[INFO] Normalizing each band to be from 0 to 255...")
    normalize_stack = normalize_stack_0_to_255(stack_filtered)

    return normalize_stack, band_names_filtered