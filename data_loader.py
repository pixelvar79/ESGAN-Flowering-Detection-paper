# import numpy as np
# import pandas as pd
# from tifffile import imread
# from skimage.transform import resize
# from pathlib import Path
# import os

# def load_image(picture):
    
#     img = imread(picture)
#     img = resize(img, (72, 72, 9))
        
#     return img


# def rescale(arr):
#     arr_min = arr.min()
#     arr_max = arr.max()
#     return (arr - arr_min) / (arr_max - arr_min)


# def load_datasett(img_dir, gt_dir):
#     csv_file = Path(gt_dir) / 'ground_truth.csv'
#     # Ensure 'concdate' is loaded as text
#     df = pd.read_csv(csv_file)
    
#     df = df[df['data_available_f50'] == 'yes']
#     y = df['f50_head_0904']
#     y1 = df['f50_head_0919']
#     y11 = df['f50_head_1005']
#     #floweringdate = df['f50_head_date']
#     #concdateID = df['concdate']
    
#     # Sort the plot_id column in ascending order
#     df_sorted = df.sort_values(by='plot_id')

#     # Load and rescale only the necessary images
#     img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df_sorted['plot_id']]
#     #print(f'List of images are: {img_files}') 

#     # Load images from the sorted img_files
#     img_list = [load_image(file) for file in img_files if file.exists()]
    
#     print(f'List of images are: {img_list}') 
            
#     x = np.stack(img_list)
    
#     # Slicing and stacking
#     dates = ['247', '262', '279']  # Julian dates for each image
#     # Slicing and concatenating
#     slices = [(dates[0], slice(0, 3)), (dates[1], slice(3, 6)), (dates[2], slice(6, 9))]
    
#     list_data = []
#     labels = []
#     rescaled_img = []

#     for names, slicing in slices:
#         list_data.append(x[:, :, :, slicing])
#         print(f'slicing and concatenating the data')
#         labels.append(names)
        
#     x1 = np.concatenate([list_data[0], list_data[1], list_data[2]], 0)
        
#     rescaled_img = [rescale(im) for im in x1]
#     x11 = np.stack(rescaled_img, axis=0)
    
#     # Print dimensions of x before slicing
#     print(f'x dimensions: {x11.shape}')

#     #floweringdate = pd.Series(np.concatenate((floweringdate, floweringdate, floweringdate), axis=0))
#     #concdateID = pd.Series(np.concatenate((concdateID, concdateID, concdateID), axis=0))
    
#     yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
    
#     # # Create the grouping column
#     # grouping = pd.Series(np.concatenate((
#     #     np.full(len(y), 247),
#     #     np.full(len(y1), 262),
#     #     np.full(len(y11), 279)
#     # ), axis=0))
    
#     return x11, yy

# import numpy as np
# import pandas as pd
# from tifffile import imread, imsave
# from skimage.transform import resize
# from pathlib import Path
# import os

# def load_image(picture):
#     img = imread(picture)
#     img = resize(img, (72, 72, 9))
#     return img

# def rescale(arr):
#     arr_min = arr.min()
#     arr_max = arr.max()
#     return (arr - arr_min) / (arr_max - arr_min)

# def load_datasett(img_dir, gt_dir, output_directory):
#     csv_file = Path(gt_dir) / 'ground_truth.csv'
#     # Ensure 'concdate' is loaded as text
#     df = pd.read_csv(csv_file)
    
#     df = df[df['data_available_f50'] == 'yes']
#     y = df['f50_head_0904']
#     y1 = df['f50_head_0919']
#     y11 = df['f50_head_1005']
    
#     # Sort the plot_id column in ascending order
#     df_sorted = df.sort_values(by='plot_id')

#     # Load and rescale only the necessary images
#     img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df_sorted['plot_id']]
#     img_list = [load_image(file) for file in img_files if file.exists()]
    
#     x = np.stack(img_list)
    
#     # Slicing and stacking
#     dates = ['247', '262', '279']  # Julian dates for each image
#     slices = [(dates[0], slice(0, 3)), (dates[1], slice(3, 6)), (dates[2], slice(6, 9))]
    
#     list_data = []
#     labels = []
#     rescaled_img = []

#     for names, slicing in slices:
#         list_data.append(x[:, :, :, slicing])
#         labels.append(names)
        
#     x1 = np.concatenate([list_data[0], list_data[1], list_data[2]], 0)
#     rescaled_img = [rescale(im) for im in x1]
#     x11 = np.stack(rescaled_img, axis=0)
    
#     # Print dimensions of x before slicing
#     print(f'x dimensions: {x11.shape}')
    
#     yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
    
#     # Create IMG subfolder in the output directory
#     img_folder = Path(output_directory) / 'IMG'
#     img_folder.mkdir(parents=True, exist_ok=True)
    
#     # Save each rescaled_img as a .tif file
#      # Save each rescaled_img as a .tif file
#     plot_ids = []
#     for i, img in enumerate(rescaled_img):
#         plot_id = f'{i+1}'
#         plot_ids.append(plot_id)
#         img_path = img_folder / plot_id
#         imsave(img_path, img)
    
#      # Save yy and plot_id as a CSV file
#     csv_path = Path(output_directory) / 'yy.csv'
#     yy_df = pd.DataFrame({'plot_id': plot_ids, 'yy': yy})
#     yy_df.to_csv(csv_path, index=False)
    
#     return x11, yy

import numpy as np
import pandas as pd
from tifffile import imread, imsave
from skimage.transform import resize
from pathlib import Path
import os

def load_image(picture):
    img = imread(picture)
    img = resize(img, (72, 72, 3))
    return img

def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_max)

def load_dataset(img_dir, gt_dir):
    csv_file = Path(gt_dir) / 'ground_truth_final.csv'
    # Ensure 'concdate' is loaded as text
    df = pd.read_csv(csv_file)
    
    # Sort the plot_id column in ascending order
    df_sorted = df.sort_values(by='plot_id')

    # Load and rescale only the necessary images
    img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df_sorted['plot_id']]
    img_list = [load_image(file) for file in img_files if file.exists()]
    
    x = np.stack(img_list)
    
    # Rescale images
    #rescaled_img = [rescale(im) for im in x]
    x11 = np.stack(x, axis=0)
    
    # Print dimensions of x before slicing
    print(f'x dimensions: {x11.shape}')
    
    # Extract yy column
    yy = df_sorted['yy']
    
    return x11, yy


