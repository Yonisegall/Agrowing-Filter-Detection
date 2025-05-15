import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.widgets import Button

import cv2
from pathlib import Path
from helper_functions.detect_edges import detect_edges

project_root = Path(__file__).resolve().parents[2]
image_npz_files = [

    r'Colors_normalize3.npz'
]

graph_f_paths = [(project_root / 'stacks' / path) for path in image_npz_files]
show_edges = False

# Load all data
graphs_data = [np.load(path)['stack'] for path in graph_f_paths]
bands = np.load(graph_f_paths[0])['band_names']
assert all(graph_data.shape == graphs_data[0].shape for graph_data in graphs_data), "Mismatched data shapes."
assert all(np.all(np.load(path)['band_names'] == bands) for path in graph_f_paths), "Band mismatch."

print("Loaded data.")

# Normalize for display
def normalize_image_stack(stack):
    norm_stack = np.empty_like(stack, dtype=np.uint8)
    for i in range(stack.shape[2]):
        band = stack[:, :, i]
        band_normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
        norm_stack[:, :, i] = band_normalized.astype(np.uint8)
    return norm_stack

# Precompute normalized versions for display
display_stacks = []
if show_edges:
    display_stacks = [np.stack([detect_edges(graph[:, :, i]) for i in range(graph.shape[2])], axis=2)
                      for graph in graphs_data]
else:
    display_stacks = [normalize_image_stack(graph) for graph in graphs_data]

graph_names = [file.replace('.npz', '') for file in image_npz_files]
current_image_index = [0]
current_band_index = [0]


fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(2, 3, height_ratios=[0.95, 0.05], width_ratios=[1.2, 1.2, 1])
ax_img = fig.add_subplot(gs[0, 0])
ax_graph = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[0, 2])
ax_table.axis("off")


img = ax_img.imshow(display_stacks[current_image_index[0]][:, :, current_band_index[0]], cmap='gray')
title = ax_img.set_title(f"{graph_names[current_image_index[0]]} - Wavelength: {bands[current_band_index[0]]} nm")


# Get position of ax_img (left, bottom, width, height)

bbox = ax_img.get_position()

num_buttons = len(bands)
button_height = 0.03
button_width = bbox.width / num_buttons

button_axes = []
buttons = []

def select_band(index):
    current_band_index[0] = index
    update_display()

for i, band in enumerate(bands):
    x_pos = (bbox.x0 - 0.082) + (i * (button_width + 0.0055))
    y_pos = bbox.y0 - 0.25

    ax_button = fig.add_axes([x_pos, y_pos, button_width, button_height])
    btn = Button(ax_button, f'{band}')
    btn.on_clicked(lambda event, idx=i: select_band(idx))
    button_axes.append(ax_button)
    buttons.append(btn)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def update_display():
    band = current_band_index[0]
    img_idx = current_image_index[0]
    img.set_data(display_stacks[img_idx][:, :, band])
    title.set_text(f"{graph_names[img_idx]} - Wavelength: {bands[band]} nm")
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == 'right' and current_image_index[0] < len(display_stacks) - 1:
        current_image_index[0] += 1
        update_display()
    elif event.key == 'left' and current_image_index[0] > 0:
        current_image_index[0] -= 1
        update_display()
    elif event.key == 'up' and current_band_index[0] < len(bands) - 1:
        current_band_index[0] += 1
        update_display()
    elif event.key == 'down' and current_band_index[0] > 0:
        current_band_index[0] -= 1
        update_display()


def on_hover(event):
    if event.inaxes == ax_img and event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        ax_table.clear()
        ax_graph.clear()
        ax_table.axis("off")

        values_all_images = [graph[y, x, :] for graph in graphs_data]
        base = values_all_images[0]

        # Spectral profiles
        for i, values in enumerate(values_all_images):
            ax_graph.plot(bands, values, label=graph_names[i], color=color_cycle[i % len(color_cycle)])
        for i in range(1, len(values_all_images)):
            diff = values_all_images[i] - base
            div = np.divide(values_all_images[i], base, out=np.full(base.shape, np.nan, dtype=np.float64), where=base != 0)
            ax_graph.plot(bands, diff, '--', label=f"Diff: {graph_names[i]}", color=color_cycle[i % len(color_cycle)])
            ax_graph.plot(bands, div, ':', label=f"Div: {graph_names[i]}", color=color_cycle[i % len(color_cycle)])

        ax_graph.set_title("Spectral Profile")
        ax_graph.set_xlabel("Wavelength (nm)")
        ax_graph.set_ylabel("Value")
        ax_graph.grid(True)
        ax_graph.legend(fontsize=8)

        # Table
        table_data = []
        header = ["Band (nm)"] + graph_names + [f"Diff {i}" for i in range(1, len(graph_names))] + [f"Div {i}" for i in range(1, len(graph_names))]
        for i, band in enumerate(bands):
            row = [f"{band}"]
            for values in values_all_images:
                v = values[i]
                row.append(f"{round(v) if abs(v) >= 2 else round(v * 10) / 10}")
            for j in range(1, len(values_all_images)):
                d = values_all_images[j][i] - base[i]
                row.append(f"{round(d) if abs(d) >= 2 else round(d * 10) / 10}")
            for j in range(1, len(values_all_images)):
                b = base[i]
                v = values_all_images[j][i]
                div = v / b if b != 0 else np.nan
                row.append(f"{round(div * 10) / 10 if not np.isnan(div) else 'NaN'}")
            table_data.append(row)

        the_table = ax_table.table(
            cellText=table_data,
            colLabels=header,
            loc='center',
            cellLoc='center'
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.auto_set_column_width(col=list(range(len(header))))

        fig.canvas.draw_idle()


fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('motion_notify_event', on_hover)
plt.tight_layout()
plt.show()


