import matplotlib
import numpy as np
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from pathlib import Path
from helper_functions.hyperspectral_to_rgb import hyperspectral_to_rgb
from helper_functions.material_color_options import COLOR_OPTIONS, get_contrasting_text_color


# Predefined list of colors to choose for materials

class MaterialSelectorApp:
    def __init__(self, root, stack, band_names):
        self.root = root
        self.stack = stack  # Hyperspectral data: H x W x B
        self.rgb_image = hyperspectral_to_rgb(stack, band_names)
        self.materials = []  # List of {'material', 'color', 'pixels', 'spectra'}
        self.current_material = None

        # Set up UI
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Hyperspectral Material Selector")
        # Main frame
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Left: Matplotlib canvas
        fig, ax = plt.subplots()
        ax.imshow(self.rgb_image)
        ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Lasso setup
        self.lasso = None

        # Right: Controls
        ctrl_frame = tk.Frame(frame)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(ctrl_frame, text="New Material", command=self.new_material).pack(pady=5)
        tk.Button(ctrl_frame, text="Finish Material", command=self.finish_material).pack(pady=5)
        tk.Button(ctrl_frame, text="Save", command=self.save_materials).pack(pady=5)

        sep = ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=5)

        # Materials list
        self.mat_list_frame = tk.Frame(ctrl_frame)
        self.mat_list_frame.pack(fill=tk.BOTH, expand=True)

    def new_material(self):
        if self.current_material:
            messagebox.showwarning("Unfinished Material", "Please finish the current material first.")
            return
        # Prompt for material
        material = simpledialog.askstring("Material Name", "Enter material name:", parent=self.root)
        if not material:
            return
        color = self.choose_color()
        # Initialize current material
        self.current_material = {'material': material, 'color': color, 'pixels': [], 'spectra': []}
        # Activate lasso selector
        self.start_lasso()

    def choose_color(self):
        # Create color selection window
        color_win = tk.Toplevel(self.root)
        color_win.title("Choose Color")

        tk.Label(color_win, text="Select a color:").pack(pady=5)

        selected_color = tk.StringVar(value=None)

        radio_frame = tk.Frame(color_win)
        radio_frame.pack(pady=5)

        for color in COLOR_OPTIONS:
            fg = get_contrasting_text_color(color)
            rb = tk.Radiobutton(
                radio_frame,
                text=color,
                variable=selected_color,
                value=color,
                bg=color,
                fg=fg,
                indicatoron=0,
                width=15,
                relief="ridge",
                bd=2
            )
            rb.pack(anchor='w', pady=2)

        def on_ok():
            color_win.destroy()

        tk.Button(color_win, text="OK", command=on_ok).pack(pady=5)
        self.root.wait_window(color_win)

        return selected_color.get()

    def start_lasso(self):
        ax = self.canvas.figure.axes[0]
        messagebox.showinfo("Selection", "Draw a polygon around the area of the material, then close the polygon to confirm.")
        self.lasso = LassoSelector(ax, onselect=self.on_lasso_select)

    def on_lasso_select(self, verts):
        # # Disable selector
        # if self.lasso:
        #     self.lasso.disconnect_events()
        #     self.lasso = None
        path = matplotlib.path.Path(verts)
        # Create grid of pixel coordinates
        h, w = self.stack.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        coords = np.vstack((xs.ravel(), ys.ravel())).T  # (N,2) as (x, y)
        mask = path.contains_points(coords).reshape((h, w))
        rows, cols = np.nonzero(mask)


        # Add newly selected pixels and spectra
        new_pixels = list(zip(rows.tolist(), cols.tolist()))
        self.current_material['pixels'].extend(new_pixels)
        new_spectra = self.stack[rows, cols, :].tolist()
        self.current_material['spectra'].extend(new_spectra)

    def update_material_list(self):
        # Clear frame
        for child in self.mat_list_frame.winfo_children():
            child.destroy()
        # Add each material
        for idx, mat in enumerate(self.materials):
            row = tk.Frame(self.mat_list_frame)
            row.pack(fill=tk.X, pady=2)
            color_lbl = tk.Label(row, width=2, bg=mat['color'])
            color_lbl.pack(side=tk.LEFT, padx=2)
            material_lbl = tk.Label(row, text=mat['material'])
            material_lbl.pack(side=tk.LEFT, padx=5)
            del_btn = tk.Button(row, text="Delete", command=lambda i=idx: self.delete_material(i))
            del_btn.pack(side=tk.RIGHT, padx=2)

    def finish_material(self):
        if self.current_material:
            self.materials.append(self.current_material)
            self.current_material = None
            self.update_material_list()

    def delete_material(self, index):
        del self.materials[index]
        self.update_material_list()


    def save_materials(self):
        if not self.materials:
            messagebox.showwarning("No Data", "No materials to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')]
        )
        if not file_path:
            return
        # Prepare JSON-serializable data
        # todo: save band names as well
        out = []
        for mat in self.materials:
            out.append({
                'material': mat['material'],
                'color': mat['color'],
                'pixels': mat['pixels'],
                'spectra': mat['spectra']
            })
        import json
        with open(file_path, 'w') as f:
            json.dump(out, f)
        messagebox.showinfo("Saved", f"Materials saved to {file_path}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    npz_file_path = project_root / r'stacks\im020.npz'
    stack = np.load(npz_file_path)['stack']
    band_names = np.load(npz_file_path)['band_names']

    root = tk.Tk()
    app = MaterialSelectorApp(root, stack, band_names)
    root.mainloop()
