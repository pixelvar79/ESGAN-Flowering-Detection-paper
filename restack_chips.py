
import os
import rasterio
import numpy as np
import cv2

# Define input and output directories
input_dir = '../images'
output_dir = '../images11'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def list_tiff_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.tif')]

# List TIFF files in the input directory
tiff_files = list_tiff_files(input_dir)

# Slices to keep
#selected_slices = [slice(61, 64), slice(67, 70), slice(73, 76), slice(79,82)]
selected_slices = [slice(61, 64), slice(67, 70), slice(73, 76), slice(79,82)]
#discard_slices = [slice(25, 31)]

# Function to resize image using OpenCV
def resize_image(image, new_height, new_width):
    resized_image = np.empty((image.shape[0], new_height, new_width), dtype=image.dtype)
    for i in range(image.shape[0]):
        resized_image[i] = cv2.resize(image[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

# Function to select specified slices from the depth dimension
def selected_slices_from_image(image, selected_slices):
    selected_images = [image[s] for s in selected_slices]
    return np.concatenate(selected_images, axis=0)

# Process each TIFF file
for tiff_file in tiff_files:
    input_path = os.path.join(input_dir, tiff_file)
    output_path = os.path.join(output_dir, tiff_file)
    
    with rasterio.open(input_path) as src:
        # Read the image
        image = src.read()
        
        # Resize the image to 108x107
        resized_image = resize_image(image, 72, 72)
        
        # Discard the specified slices
        modified_image = selected_slices_from_image(resized_image, selected_slices)
        
        # Save the modified image
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=modified_image.shape[1],
            width=modified_image.shape[2],
            count=modified_image.shape[0],
            dtype=modified_image.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(modified_image)