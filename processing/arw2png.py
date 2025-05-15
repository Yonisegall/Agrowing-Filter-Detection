import os
import subprocess

def convert_arw_to_png(input_path: str, output_path: str) -> None:
    """
    Converts an .arw file to a .png file using ImageMagick.

    Args:
        input_path (str): The file path to the input .arw file.
        output_path (str): The file path for the output .png file.
    """
    try:
        # Run the ImageMagick convert command
        subprocess.run(["magick", input_path, output_path], check=True)
        print(f"Conversion successful: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

# Example usage
image_arw = r'data/M24_23_03_25_TDP00872.ARW'
image_png =  f"{os.path.splitext(image_arw)[0]}.png"  # Split the file path and extension
convert_arw_to_png(image_arw, image_png)
