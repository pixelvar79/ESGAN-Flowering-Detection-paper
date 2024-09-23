import os

# Set base directory as current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Set the base directory one level up
base_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Define subdirectories relative to the base directory
dir_img = os.path.join(base_dir, 'images1')
dir_img1 = os.path.join(base_dir, 'images11')
dir_gt = os.path.join(base_dir, 'groundtruth')

dir_out = os.path.join(base_dir, 'output', 'predictions_09222024')
# Create the output directory if it does not exist
os.makedirs(dir_out, exist_ok=True)

