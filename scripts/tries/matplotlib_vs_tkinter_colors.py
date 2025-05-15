import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

from helper_functions.material_color_options import COLOR_OPTIONS


def create_matplotlib_figure(colors):
    fig, ax = plt.subplots(figsize=(4, len(colors) * 0.35), dpi=100)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, len(colors))
    ax.axis('off')

    for i, color in enumerate(colors):
        rect = mpatches.Rectangle((0.5, i), 2, 0.8, color=color)
        ax.add_patch(rect)
        ax.text(3, i + 0.4, color, va='center', fontsize=10)

    fig.tight_layout()
    return fig

def show_combined_preview(colors):
    root = tk.Tk()
    root.title("Matplotlib & Tkinter Color Preview")

    # --- Tkinter Color Preview (left side) ---
    tk_frame = tk.Frame(root)
    tk_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    tk.Label(tk_frame, text="Tkinter Preview", font=('Arial', 12, 'bold')).pack(pady=(0, 5))

    for color in colors:
        frame = tk.Frame(tk_frame, bg=color, width=120, height=25)
        frame.pack_propagate(False)
        label = tk.Label(
            frame,
            text=color,
            bg=color,
            fg='white' if color not in ('white', 'yellow', 'lime', 'gold') else 'black'
        )
        label.pack(fill=tk.BOTH, expand=True)
        frame.pack(padx=5, pady=1)

    # --- Matplotlib Color Preview (right side in canvas) ---
    mpl_frame = tk.Frame(root)
    mpl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    tk.Label(mpl_frame, text="Matplotlib Preview", font=('Arial', 12, 'bold')).pack(pady=(0, 5))

    fig = create_matplotlib_figure(colors[::-1])
    canvas = FigureCanvasTkAgg(fig, master=mpl_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()

# Show combined preview
show_combined_preview(COLOR_OPTIONS)
