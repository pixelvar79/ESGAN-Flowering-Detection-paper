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

def load_image1(picture):
    
    img = imread(picture)
    img = resize(img, (72, 72, 12))
        
    return img

def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)



def load_datasett(img_dir, gt_dir):
    csv_file = Path(gt_dir) / 'ground_truth.csv'
    # Ensure 'concdate' is loaded as text
    df = pd.read_csv(csv_file, dtype={'concdate': str})
    
    df = df[df['data_available_f50'] == 'yes']
    y = df['f50_head_0904']
    y1 = df['f50_head_0919']
    y11 = df['f50_head_1005']
    floweringdate = df['f50_head_date']
    concdateID = df['concdate']
    
    # Sort the plot_id column in ascending order
    df_sorted = df.sort_values(by='plot_id')

    # Load and rescale only the necessary images
    img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df_sorted['plot_id']]

    # Load images from the sorted img_files
    img_list = [load_image(file) for file in img_files if file.exists()]
            
    x = np.stack(img_list)
    
    # Slicing and stacking
    dates = ['247', '262', '279']  # Julian dates for each image
    # Slicing and concatenating
    slices = [(dates[0], slice(0, 3)), (dates[1], slice(3, 6)), (dates[2], slice(6, 9))]
    
    list_data = []
    labels = []
    rescaled_img = []

    for names, slicing in slices:
        list_data.append(x[:, :, :, slicing])
        print(f'slicing and concatenating the data')
        labels.append(names)
        
    x1 = np.concatenate([list_data[0], list_data[1], list_data[2]], 0)
        
    rescaled_img = [rescale(im) for im in x1]
    x11 = np.stack(rescaled_img, axis=0)
    
    # Print dimensions of x before slicing
    print(f'x dimensions: {x11.shape}')

    floweringdate = pd.Series(np.concatenate((floweringdate, floweringdate, floweringdate), axis=0))
    concdateID = pd.Series(np.concatenate((concdateID, concdateID, concdateID), axis=0))
    
    yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
    
    # Create the grouping column
    grouping = pd.Series(np.concatenate((
        np.full(len(y), 247),
        np.full(len(y1), 262),
        np.full(len(y11), 279)
    ), axis=0))
    
    return x11, yy, floweringdate, concdateID, grouping

def load_datasettt(img_dir, gt_dir):
    csv_file = Path(gt_dir) / 'MSA_GT.csv'
    # Ensure 'concdate' is loaded as text
    df = pd.read_csv(csv_file, dtype={'concdate': str})
    
    df = df[df['data_available_f50'] == 'yes']
    y = df['f50_head_0904']
    y1 = df['f50_head_0919']
    y11 = df['f50_head_1005']
    y111 = df['f50_head_1022']
    floweringdate = df['f50_head_date']
    concdateID = df['concdate']
    
    # Sort the plot_id column in ascending order
    df_sorted = df.sort_values(by='plot_id')

    # Load and rescale only the necessary images
    img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df_sorted['plot_id']]

    # Load images from the sorted img_files
    img_list = [load_image1(file) for file in img_files if file.exists()]
            
    x = np.stack(img_list)
    
    # Slicing and stacking
    dates = ['247', '262', '279', '296']  # Julian dates for each image
    # Slicing and concatenating
    slices = [(dates[0], slice(0, 3)), (dates[1], slice(3, 6)), (dates[2], slice(6, 9)), (dates[3], slice(9, 12))]
    
    list_data = []
    labels = []
    rescaled_img = []

    for names, slicing in slices:
        list_data.append(x[:, :, :, slicing])
        print(f'slicing and concatenating the data')
        labels.append(names)
        
    x1 = np.concatenate([list_data[0], list_data[1], list_data[2], list_data[3]],0)
        
    rescaled_img = [rescale(im) for im in x1]
    x11 = np.stack(rescaled_img, axis=0)
    
    # Print dimensions of x before slicing
    print(f'x dimensions: {x11.shape}')

    floweringdate = pd.Series(np.concatenate((floweringdate, floweringdate, floweringdate, floweringdate), axis=0))
    concdateID = pd.Series(np.concatenate((concdateID, concdateID, concdateID, concdateID), axis=0))
    
    yy = pd.Series(np.concatenate((y, y1, y11, y111), axis=0))
    
    # Create the grouping column
    grouping = pd.Series(np.concatenate((
        np.full(len(y), 247),
        np.full(len(y1), 262),
        np.full(len(y11), 279),
        np.full(len(y11), 296)
    ), axis=0))
    
    return x11, yy, floweringdate, concdateID, grouping


def load_dataset(img_dir, gt_dir):
        csv_file = Path(gt_dir) / 'MSA_GT.csv'
        #df = pd.read_csv(csv_file)
        # Ensure 'concdate' is loaded as text
        df = pd.read_csv(csv_file, dtype={'concdate': str})
        
        df = df[df['data_available_f50'] == 'yes']
        y = df['f50_head_0904']
        y1 = df['f50_head_0919']
        y11 = df['f50_head_1005']
        #y111 = df['f50_head_1022']
        floweringdate = df['f50_head_date']
        concdateID = df['concdate']
        
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

        floweringdate = pd.Series(np.concatenate((floweringdate, floweringdate, floweringdate), axis=0))
        concdateID = pd.Series(np.concatenate((concdateID, concdateID, concdateID), axis=0))
        
        yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
        
        # Print dimensions of y
        print(f'y dimensions: {yy.shape}')
        
        #return x11, yy, floweringdate, concdateID
        return x11, yy


# def load_dataset1(img_dir, gt_dir):
#     csv_file = Path(gt_dir) / 'MSA_GT.csv'
#     #df = pd.read_csv(csv_file)
#     # Ensure 'concdate' is loaded as text
#     df = pd.read_csv(csv_file, dtype={'concdate': str})
    
#     df = df[df['data_available_f50'] == 'yes']
#     y = df['f50_head_0904']
#     y1 = df['f50_head_0919']
#     y11 = df['f50_head_1005']
#     y111 = df['f50_head_1022']
#     floweringdate = df['f50_head_date']
#     concdateID = df['concdate']
    
#     # Load and rescale only the necessary images
#     img_files = [Path(img_dir) / f'plot_{plot_id}.tif' for plot_id in df['plot_id']]
#     img_list = [load_image1(file) for file in img_files if file.exists()]
    
#     x = np.stack(img_list)
    
#     # Slicing and stacking
#     dates = ['247', '262', '279', '296']  # Julian dates for each image
#     # Slicing and concatenating
#     slices = [(dates[0],slice(0, 3)), (dates[1],slice(3, 6)), (dates[2],slice(6, 9)), (dates[3], slice(9,12))]
    
#     list_data = []
#     labels = []
#     rescaled_img =[]

#     for names, slicing in slices:
#         list_data.append(x[:, :, :, slicing])
#         print(f'slicing and concatanating the data')
#         labels.append(names)
        
#     x1 = np.concatenate([list_data[0], list_data[1], list_data[2],list_data[3]], 0)
        
#     rescaled_img = [rescale(im) for im in x1]
#     x11 = np.stack(rescaled_img, axis=0)
    
#     # Print dimensions of x before slicing
#     print(f'x dimensions: {x11.shape}')

#     floweringdate = pd.Series(np.concatenate((floweringdate, floweringdate, floweringdate, floweringdate), axis=0))
#     concdateID = pd.Series(np.concatenate((concdateID, concdateID, concdateID, concdateID), axis=0))
    
#     yy = pd.Series(np.concatenate((y, y1, y11, y111), axis=0))
    
#     # Print dimensions of y
#     print(f'y dimensions: {yy.shape}')
    
#     #return x11, yy, floweringdate, concdateID
#     return x11, yy