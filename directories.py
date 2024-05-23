import os
from tifffile import imread
from skimage.transform import resize
from pathlib import Path

# Set base directory
# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Set the base directory one level up
base_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Now you can use base_dir as your base directory
# Define subdirectories relative to the base directory
dir_img = os.path.join(base_dir, 'images')
dir_gt = os.path.join(base_dir, 'groundtruth')
dir_out = os.path.join(base_dir, 'output', 'head50')

# Print the paths to verify
print(f'The images directory is: {dir_img}')
print(f'The ground truth directory is: {dir_gt}')
print(f'The output directory is {dir_out}')

