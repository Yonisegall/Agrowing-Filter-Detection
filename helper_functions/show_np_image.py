import cv2
import numpy as np


def show_image(image, window_name):
    # can be either grayscale or rgb
    resized = cv2.resize(image, (500, 500))
    normalized = resized.astype(np.float32) / 255
    cv2.imshow(window_name, normalized)



def show_image_with_zoom(image, window_name):
    zoom_factor = 1.0
    min_zoom = 0.1
    max_zoom = 10.0

    def update_display():
        # Calculate zoomed size
        h, w = image.shape[:2]
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Normalize for display
        display = resized.astype(np.float32) / 255
        cv2.imshow(window_name, display)

    def on_key(event):
        nonlocal zoom_factor
        if event == ord('+') or event == ord('='):
            zoom_factor = min(zoom_factor * 1.1, max_zoom)
        elif event == ord('-') or event == ord('_'):
            zoom_factor = max(zoom_factor / 1.1, min_zoom)
        update_display()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow window resizing
    update_display()

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
        on_key(key)

    cv2.destroyAllWindows()
