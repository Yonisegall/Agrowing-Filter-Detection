import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class CubeStatisticsViewer:
    def __init__(self, root, stack, band_names, rgb_image, percentiles=[25, 50, 75]):
        self.root = root
        self.stack = stack
        self.band_names = band_names
        self.rgb_image = rgb_image
        self.percentiles = percentiles

        self.root.title("Hyperspectral Cube Statistics")
        self.root.geometry("1600x900")

        self.setup_layout()

    def setup_layout(self):
        # Split to image on left and graph on right
        self.main_panes = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_panes.pack(fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(self.main_panes)
        self.graph_frame = tk.Frame(self.main_panes)

        self.main_panes.add(self.image_frame, stretch="always")
        self.main_panes.add(self.graph_frame, stretch="always")

        # Create figure and axes
        self.fig = Figure(figsize=(15, 5), dpi=100)
        self.ax_img = self.fig.add_subplot(121)
        self.ax_plot = self.fig.add_subplot(122)

        self.ax_img.axis('off')
        self.ax_img.set_title("RGB Image")
        self.ax_img.imshow(self.rgb_image)

        self.plot_cube_statistics()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def plot_cube_statistics(self):
        reshaped_stack = self.stack.reshape(-1, self.stack.shape[-1])
        min_vals = reshaped_stack.min(axis=0)
        max_vals = reshaped_stack.max(axis=0)

        self.ax_plot.plot(self.band_names, min_vals, label='Minimum')
        self.ax_plot.plot(self.band_names, max_vals, label='Maximum')

        # Plot percentiles based on user input
        for p in self.percentiles:
            q = np.percentile(reshaped_stack, p, axis=0)
            self.ax_plot.plot(self.band_names, q, label=f'{p}th Percentile')

        self.ax_plot.set_title("Spectral Statistics Across All Pixels")
        self.ax_plot.set_xlabel("Wavelength (nm)")
        self.ax_plot.set_ylabel("Intensity")
        self.ax_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.ax_plot.grid(True)
