from pathlib import Path
import numpy as np
from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.stack_graph_w_percentile import CubeStatisticsViewer
import tkinter as tk

# Load data
project_root = Path(__file__).resolve().parents[2]
# npz_file_path = project_root / r'stacks\im020.npz'
# npz_file_path = project_root / r'stacks\before_changes1.npz'
# npz_file_path = project_root / r'stacks\before_changes1_unnormalize.npz'

# npz_file_path = project_root / r'stacks\Colors.npz'
npz_file_path = project_root / r'stacks\Colors_unnormalize.npz'

data = np.load(npz_file_path)
stack = data['stack']
band_names = data['band_names']
rgb_image = hyperspectral_to_rgb(stack, band_names)

# Example reference values
custom_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Launch viewer
root = tk.Tk()
app = CubeStatisticsViewer(root, stack, band_names, rgb_image, percentiles=custom_percentiles)
root.mainloop()

# percentiles=custom_percentiles