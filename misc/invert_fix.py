import os
from PIL import Image
import numpy as np

def invert_bw_images_fast(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    count = 0

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if count % 100 == 0:
            print("files inverted: " + str(count))
        count += 1
        if filename.lower().endswith('.jpg'):  # Process only .jpg files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image and convert to grayscale (if needed)
            with Image.open(input_path) as img:
                img = img.convert("L")  # Ensure grayscale

                # Convert the image to a NumPy array
                img_array = np.array(img)

                # Invert the image (255 - pixel values)
                inverted_array = 255 - img_array

                # Convert the inverted array back to an image
                inverted_img = Image.fromarray(inverted_array)

                # Save the inverted image
                inverted_img.save(output_path)

            #print(f"Inverted: {filename} -> {output_path}")


# Example usage
input_folder = "/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text_plotfiles/frames/game_30_28.1.24 CHI @ POR_720p60_NBCS-CHI.mp4/frames"
output_folder = "/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/img_to_text/test_bla"
invert_bw_images_fast(input_folder, output_folder)
