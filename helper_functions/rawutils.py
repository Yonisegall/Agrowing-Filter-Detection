import rawpy
import numpy as np
from helper_functions.stack import normalize_stack_0_to_255


def read_arw_as_rgb(path, normalize=False):
    """
    Loads a Sony .ARW RAW file and returns a postprocessed RGB image.
    
    Parameters:
        path (str): Path to the .ARW file.
        normalize (bool): If True, scales RGB to [0, 255] float32.

    Returns:
        np.ndarray: RGB image as NumPy array.
    """
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            no_auto_bright=True,
            output_bps=16,       # preserve dynamic range
            gamma=(1, 1)         # linear gamma
        )
    # with rawpy.imread(str(path)) as raw:
    #     rgb = raw.postprocess()
    rgb = rgb.astype(np.float32)
    normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255

    if normalize:
        rgb = normalize_stack_0_to_255(rgb)
    return normalized
