# import numpy as np
# import pandas as pd
# from tifffile import imread
# from skimage.transform import resize
# from pathlib import Path
# import os

# def load_image(picture):
#     print(f'image: {os.path.basename(picture)} is being loaded')
#     img = imread(picture)
#     img = resize(img, (72, 72, 96))
#     return img

# def load_dataset(img_dir, gt_dir, task='classification'):
#     #print("img_dir:", img_dir)
#     #print(Path(img_dir))
#     #print("Files in img_dir:", os.listdir(img_dir))  # Print contents of img_dir
#     img_list = [load_image(file) for file in sorted(Path(img_dir).glob('*.tif'), key=lambda x: int(x.stem.split('_')[1]))]
#     #print(img_list)
#     #print("img_list:", img_list) 
#     x = np.stack(img_list)
    
#     csv_file = Path(gt_dir) / 'MSA_GT.csv'
#     df = pd.read_csv(csv_file)
    
#     if task == 'classification2':
#         df = df[df['data_available_f50'] == 'yes']
#         y = df['f50_head_0904']
#         y1 = df['f50_head_0919']
#         y11 = df['f50_head_1005']
#         x = [x[i] for i in df.index]
#         x = np.array(x)
#     else:
#         y = df['f50_head_date']
        
#     return x, y, y1, y11

# def preparing_data(X, y, y1, y11, dates=('247', '262', '279')):
#     print(f'data is being prepared according to the input format')
    
#     # slices = ((dates[0], slice(61, 64)), (dates[1], slice(67, 70)), (dates[2], slice(73, 76)))
    
#     list_data = []
#     labels = []

#     # for names, slicing in slices:
#     #     list_data.append(X[:, :, :, slicing])
#     #     labels.append(names)
    
#     # X1 = np.concatenate([list_data[0], list_data[1], list_data[2]], 0)
    
#     def rescale(arr):
#         arr_min = arr.min()
#         arr_max = arr.max()
#         return (arr - arr_min) / (arr_max - arr_min)

#     LIST_IMG = [rescale(im) for im in X]
#     X111 = np.stack(LIST_IMG, axis=0)
    
#     yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
    
#     return X111, yy


import numpy as np
import pandas as pd
from tifffile import imread
from skimage.transform import resize
from pathlib import Path
import os

def load_image(picture):
    
    img = imread(picture)
    img = resize(img, (72, 72, 9))
        
    return img

def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def load_dataset(img_dir, gt_dir):
    csv_file = Path(gt_dir) / 'MSA_GT.csv'
    df = pd.read_csv(csv_file)
    
    df = df[df['data_available_f50'] == 'yes']
    y = df['f50_head_0904']
    y1 = df['f50_head_0919']
    y11 = df['f50_head_1005']
    
    # Load and rescale only the necessary images
    img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df['plot_id']]
    img_list = [load_image(file) for file in img_files if file.exists()]
    
    x = np.stack(img_list)
    
    # Slicing and stacking
    dates = ['247', '262', '279']  # Julian dates for each image
    # Slicing and concatenating
    slices = [(dates[0],slice(0, 3)), (dates[1],slice(3, 6)), (dates[2],slice(6, 9))]
    
    list_data = []
    labels = []
    rescaled_img =[]

    for names, slicing in slices:
        list_data.append(x[:, :, :, slicing])
        print(f'slicing and concatanating the data')
        labels.append(names)
        
    x1 = np.concatenate([list_data[0], list_data[1], list_data[2]], 0)
        
    rescaled_img = [rescale(im) for im in x1]
    x11 = np.stack(rescaled_img, axis=0)
    
    # Print dimensions of x before slicing
    print(f'x dimensions: {x11.shape}')

    yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
    
    # Print dimensions of y
    print(f'y dimensions: {yy.shape}')
    
    return x11, yy