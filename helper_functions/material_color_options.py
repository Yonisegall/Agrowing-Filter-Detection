import matplotlib.colors as mcolors
import tkinter as tk

ORIGINAL_COLOR_OPTIONS = [
    'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
    'orange', 'purple', 'pink', 'brown', 'gray',
    'black', 'white', 'lightgray',
    'gold', 'lime', 'navy', 'teal', 'maroon', 'olive',
    'turquoise', 'violet', 'indigo', 'coral', 'crimson',
    'palegreen', 'peru', 'darkolivegreen'
]

def sort_colors_by_hue_with_sorted_grays_last(color_names):
    """
    Sorts color names by hue (in HSV), placing low-saturation colors (grays/neutrals)
    at the end, sorted by brightness.

    Parameters:
        color_names (list of str): List of color names compatible with Matplotlib.

    Returns:
        list of str: Sorted list, with colorful hues first and sorted grays/neutrals last.
    """
    def to_hsv(color_name):
        rgb = mcolors.to_rgb(color_name)
        return mcolors.rgb_to_hsv(rgb)

    colorful = []
    grays = []

    for color in color_names:
        h, s, v = to_hsv(color)
        if s < 0.2:  # grayscale or low-saturation
            grays.append((color, v))  # sort grays by brightness
        else:
            colorful.append((color, h))  # sort colorful by hue

    colorful_sorted = sorted(colorful, key=lambda x: x[1])
    grays_sorted = sorted(grays, key=lambda x: x[1])

    return [c[0] for c in colorful_sorted] + [c[0] for c in grays_sorted]

COLOR_OPTIONS = sort_colors_by_hue_with_sorted_grays_last(ORIGINAL_COLOR_OPTIONS)


def hex_to_rgb(color_str):
    """Convert a color name or hex code to an (R, G, B) tuple."""
    try:
        # Use the colorchooser to get the RGB value from the system
        # Temporarily create a color widget to get RGB values
        root = tk.Tk()
        root.withdraw()
        rgb_tuple = root.winfo_rgb(color_str)
        root.destroy()
        return tuple(c // 256 for c in rgb_tuple)
    except:
        return (0, 0, 0)  # fallback for invalid colors

def get_contrasting_text_color(bg_color):
    """Return 'black' or 'white' depending on which contrasts better with the background color."""
    r, g, b = hex_to_rgb(bg_color)
    luminance = 0.299*r + 0.587*g + 0.114*b
    return 'black' if luminance > 128 else 'white'