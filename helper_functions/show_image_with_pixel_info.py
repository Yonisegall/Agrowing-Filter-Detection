import cv2


def show_image_with_pixel_info(image):

    # Mouse callback function to display RGB values
    def show_pixel_value(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Get the BGR values from the image at (x, y)
            b, g, r = image[y, x]
            text = f"R: {r}, G: {g}, B: {b}"
            # Clone the original image to avoid overwriting
            image_part_with_text = image.copy()
            cv2.putText(image_part_with_text, text, (10, 30), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Image", image_part_with_text)

    # Create a named window and set the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", show_pixel_value)
    cv2.imshow("Image", image)
    while True:
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()