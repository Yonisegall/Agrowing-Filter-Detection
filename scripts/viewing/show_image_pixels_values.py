from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
file_path = project_root / 'stacks' / 'Colors_normalize3.npz'
data = np.load(str(file_path))

array = data[data.files[0]]
print("Shape:", array.shape)

R, G, B = 11, 6, 2

RGB_image = np.stack([array[:, :, R], array[:, :, G], array[:, :, B]], axis=-1)
RGB_image = np.clip(RGB_image, 0, 255).astype(np.uint8)

fig, ax = plt.subplots()
cax = ax.imshow(RGB_image)

def onclick(event):

    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Pixel at ({y}, {x}): {array[y, x, :]}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
