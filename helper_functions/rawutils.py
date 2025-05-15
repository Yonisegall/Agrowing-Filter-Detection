import rawpy
import numpy as np
from helper_functions.stack import normalize_stack_0_to_255


def ARW_as_Matrix(path, normalize1=False, normalize2=False, normalize3=False, normalize4=True):
    """
    Loads a Sony .ARW RAW file and returns a postprocessed RGB image.
    
    Parameters:
        path (str): Path to the .ARW file.
        normalize (bool): If True, scales RGB to [0, 255] float32.

    Returns:
        np.ndarray: RGB image as NumPy array.
    """
    with rawpy.imread(str(path)) as raw:

        RGB_image = raw.postprocess(
            use_camera_wb=False,
            no_auto_bright=True,
            output_bps=16,       # preserve dynamic range
            gamma=(1, 1)         # linear gamma
        )

    RGB_image = RGB_image.astype(np.float32)

    print(type(RGB_image))

    if normalize1:
        RGB_image = (RGB_image - RGB_image.min()) / (RGB_image.max() - RGB_image.min()) * 255

    elif normalize2:
        RGB_image  = (RGB_image / 65535.0) * 255

    elif normalize3:
        RGB_image = normalize_stack_0_to_255(RGB_image)

    elif normalize4:
        RGB_image = RGB_image/211.4

    return RGB_image
