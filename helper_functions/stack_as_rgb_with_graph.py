import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



class PixelInspectorApp:
    def __init__(self, root, stack, band_names, rgb_image, normalize_spectrum):
        self.root = root
        self.stack = stack
        self.band_names = band_names
        self.rgb_image = rgb_image
        self.normalize_spectrum = normalize_spectrum  # Flag for normalization

        self.selected_lines = []
        self.pixel_entries = []

        self.root.title("Hyperspectral Pixel Inspector")
        self.root.geometry("1600x900")

        self.setup_layout()

    def setup_layout(self):
        # ===== Create frames =====
        self.main_panes = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_panes.pack(fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(self.main_panes)
        self.graph_frame = tk.Frame(self.main_panes)
        self.ui_frame = tk.Frame(self.main_panes, width=200)

        self.main_panes.add(self.image_frame, stretch="always")
        self.main_panes.add(self.graph_frame, stretch="always")
        self.main_panes.add(self.ui_frame)

        self.ui_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ===== Set up matplotlib figure =====
        self.fig = Figure(figsize=(15, 5), dpi=100)
        self.ax_img = self.fig.add_subplot(121)
        self.ax_plot = self.fig.add_subplot(122)
        self.ax_img.axis('off')
        self.ax_img.set_title("RGB Image")
        self.ax_plot.set_title("Pixel Spectrum")
        self.ax_plot.set_xlabel("Wavelength (nm)")
        self.ax_plot.set_ylabel("Intensity")
        self.ax_plot.grid(True)

        self.ax_img.imshow(self.rgb_image)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # ===== Create the container frame for the scrollable list and the button =====
        self.pixel_list_frame = tk.Frame(self.ui_frame)
        self.pixel_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== Scrollable Pixel List =====
        self.scroll_canvas = tk.Canvas(self.pixel_list_frame, borderwidth=0)
        self.scroll_frame = tk.Frame(self.scroll_canvas)
        self.scrollbar = tk.Scrollbar(self.pixel_list_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.scroll_frame.bind("<Configure>",
                               lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        # ===== Clear All Button =====
        self.clear_button = tk.Button(self.ui_frame, text="Clear All", command=self.clear_all, bg="red", fg="white")
        self.clear_button.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Ensure the UI adjusts correctly with dynamic content
        self.ui_frame.pack_propagate(False)  # Prevent ui_frame from resizing to fit its children

    def on_click(self, event):
        if event.inaxes != self.ax_img:
            return

        x, y = int(event.xdata), int(event.ydata)
        if y < 0 or y >= self.stack.shape[0] or x < 0 or x >= self.stack.shape[1]:
            return

        spectrum = self.stack[y, x, :]

        # Normalize spectrum by making the average intensity = 1, if the flag is True
        if self.normalize_spectrum:
            mean_spectrum = np.mean(spectrum)
            if mean_spectrum != 0:  # Avoid division by zero
                spectrum = spectrum / mean_spectrum

        color = self.rgb_image[y, x, :].astype(float) / 255.0
        label = f"({x},{y})"

        # Plot on graph
        line, = self.ax_plot.plot(self.band_names, spectrum, color=color, marker='o')
        self.selected_lines.append(line)

        # UI entry
        entry_frame = tk.Frame(self.scroll_frame)
        entry_frame.pack(fill='x', pady=1)

        color_box = tk.Canvas(entry_frame, width=20, height=20)
        color_box.create_rectangle(0, 0, 20, 20, fill=self.rgb_to_hex(color), outline="")
        color_box.pack(side=tk.LEFT, padx=2)

        label_widget = tk.Label(entry_frame, text=label)
        label_widget.pack(side=tk.LEFT)

        delete_button = tk.Button(entry_frame, text="✖", fg='white', bg='black', width=3, command=lambda: self.delete_pixel(entry_frame, line))
        delete_button.pack(side=tk.RIGHT, padx=2)

        self.pixel_entries.append((entry_frame, line))

        self.canvas.draw()

    def delete_pixel(self, entry_frame, line):
        entry_frame.destroy()
        if line in self.selected_lines:
            line.remove()
            self.selected_lines.remove(line)
            self.canvas.draw()

    def clear_all(self):
        for entry_frame, line in self.pixel_entries:
            entry_frame.destroy()
            if line in self.selected_lines:
                line.remove()
        self.pixel_entries.clear()
        self.selected_lines.clear()
        self.canvas.draw()

    def rgb_to_hex(self, color):
        r, g, b = [int(c * 255) for c in color]
        return f'#{r:02x}{g:02x}{b:02x}'