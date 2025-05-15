
def split_image_to_tiles(image, cols=3, rows=2):
    """
    Splits the image into tiles ordered top-to-bottom first, then left-to-right.

    For a 3×2 grid, the order will be:
        0: top-left
        1: bottom-left
        2: top-middle
        3: bottom-middle
        4: top-right
        5: bottom-right

    Returns:
        List of tiles in the above order.
    """
    h, w = image.shape[:2]
    tile_h = h // rows
    tile_w = w // cols
    tiles = []

    for col in range(cols):        # left to right
        for row in range(rows):    # top to bottom
            x_start = col * tile_w
            y_start = row * tile_h
            tile = image[y_start:y_start + tile_h, x_start:x_start + tile_w]
            tiles.append(tile)

    return tiles


