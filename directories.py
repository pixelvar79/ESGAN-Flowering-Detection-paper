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
print(dir_img)
print(dir_gt)
print(dir_out)

# Check if output directory exists, create it if not
# if not os.path.exists(dir_out):
#     os.makedirs(dir_out)
#dir_restmodels = os.path.join(base_dir, 'Data\\GANS_FL\\output\\head50\\smallcnn')

# print(dir_out)
# print(dir_img)
# print(dir_gt)

# Function to get list of TIFF files in a directory
# def get_tiff_files(directory):
#     tiff_files = [f for f in os.listdir(directory) if f.endswith('.tif') or f.endswith('.tiff')]
#     return tiff_files

# def load_image(picture):
#     img = imread(picture)
#     img = resize(img, (72, 72, 96))
#     return img

# def load_dataset(img_dir, gt_dir, task='classification'):
#     img_list = [load_image(file) for file in sorted(Path(img_dir).glob('*.tif'), key=lambda x: int(x.stem.split('_')[1]))]
#     print(img_list)
# # # Get list of TIFF files in dir_img
# # tiff_files = get_tiff_files(dir_img)

# # # Print the list of TIFF files
# # print("Available TIFF images in dir_img:")
# # for file in tiff_files:
# #     print(file)

# # Load data
# X, y, y1, y11 = load_dataset(dir_img, dir_gt)