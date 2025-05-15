from matplotlib import pyplot as plt
import numpy as np

def show_arw(image, title="ARW Image"):
    """
    Displays an image exactly as it is, without manipulation.

    Parameters:
        image (np.ndarray): The image to display.
        title (str): The title of the plot.
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
