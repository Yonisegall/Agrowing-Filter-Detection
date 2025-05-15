from pathlib import Path
import numpy as np
from matplotlib import use as mpl_use
from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.stack_as_rgb_with_graph import PixelInspectorApp
import tkinter as tk
mpl_use("TkAgg")  # Required for Tkinter back-end


# === Load data ===
project_root = Path(__file__).resolve().parents[2]
# npz_file_path = project_root / r'stacks\try_change_det.npz'
npz_file_path = project_root / r'stacks\im020.npz'
# npz_file_path = project_root / r'stacks\im1.npz'
# npz_file_path = project_root / r'stacks\before_changes1.npz'
normalize_spectrum = False
# todo: make normalization with a button press

data = np.load(npz_file_path)
stack = data['stack']   # shape: (H, W, 14)
band_names = data['band_names']  # shape: (14,)
rgb_image = hyperspectral_to_rgb(stack, band_names)
root = tk.Tk()
app = PixelInspectorApp(root, stack, band_names, rgb_image, normalize_spectrum)
root.mainloop()
