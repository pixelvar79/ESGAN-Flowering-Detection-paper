
# # Load images and corresponding ground truth
# from directories import dir_img, dir_img1, dir_gt, dir_out
# from data_loader import load_dataset1, load_datasett, load_dataset
# from other_utils import subset_split, subset_split1, store_roc_results, summary_metrics 
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from pathlib import Path
# import os
# from tensorflow.keras.models import load_model
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from skimage.transform import resize as skimage_resize
# import numpy as np
# import tensorflow as tf
# import joblib

# # Load dataset
# X, y, floweringdate, concdateID = load_datasett(dir_img, dir_gt)

# models_dir = '../output/bestmodels/benchmarks'
# models = {}

# # Iterate through files in the models directory
# for file_name in os.listdir(models_dir):
#     file_path = os.path.join(models_dir, file_name)
    
#     # Load .pkl models using joblib
#     if file_name.endswith('.pkl'):
#         models[file_name] = joblib.load(file_path)
    
#     # Load .h5 models using Keras load_model
#     elif file_name.endswith('.h5'):
#         models[file_name] = load_model(file_path)

# # Print the loaded models
# print("Loaded models:")
# for model_name, model in models.items():
#     print(f"{model_name}: {model}")
    

# def calculate_tabular_stats(X):
#     statslist = []
#     for p in X:
#         # Calculate median for each channel
#         bmean = np.median(p[:,:,0])
#         gmean = np.median(p[:,:,1])
#         rmean = np.median(p[:,:,2])
        
#         # Calculate 99th percentile for each channel
#         bperc99 = np.percentile(p[:,:,0], 99)
#         gperc99 = np.percentile(p[:,:,1], 99)
#         rperc99 = np.percentile(p[:,:,2], 99)
        
#         # Calculate range (max - min) for each channel
#         brange = np.ptp(p[:,:,0])
#         grange = np.ptp(p[:,:,1])
#         rrange = np.ptp(p[:,:,2])
        
#         # Calculate standard deviation for each channel
#         bstd = np.std(p[:,:,0])
#         gstd = np.std(p[:,:,1])
#         rstd = np.std(p[:,:,2])
        
#         # Calculate max value for each channel
#         bmax = np.max(p[:,:,0])
#         gmax = np.max(p[:,:,1])
#         rmax = np.max(p[:,:,2])
        
#         # Create a row of statistics
#         means_row = (bmean, gmean, rmean, bperc99, gperc99, rperc99, brange, grange, rrange, bstd, gstd, rstd, bmax, gmax, rmax)
        
#         # Append the row to the stats list
#         statslist.append(means_row)

#     # Convert the list of statistics to a DataFrame
#     stats_df = pd.DataFrame(np.stack(statslist))

#     # Assign column names to the DataFrame
#     stats_df.columns = ['Blue_med', 'Green_med', 'Red_med', 'Blue_p99', 'Green_p99', 'Red_perc99', 'Blue_range',
#                         'Green_range', 'Red_range', 'bstd', 'gstd', 'rstd', 'bmax', 'gmax', 'rmax']

#     return stats_df

# # Call tabular stats generation
# stats_df = calculate_tabular_stats(X)

# def subset_split(x_var, y_var, RANDOM_STATE=123, n_classes=2):
#     # Encode the labels
#     encoder = LabelEncoder()
#     encoder.fit(y_var)
#     y_var = encoder.transform(y_var)

#     # Create a DataFrame to hold the indices
#     indices = pd.DataFrame(x_var.index)

#     # Split the data into training and testing sets
#     x_train1, x_test1, y_train1, y_test1, index_train, index_test = train_test_split(
#         x_var, y_var, indices, train_size=0.8, random_state=RANDOM_STATE
#     )

#     return x_train1, x_test1, y_train1, y_test1, index_train, index_test

# def subset_split1(x_var, y_var, RANDOM_STATE=42, n_classes=2, resize_images=False):
#     # Encode the labels
#     encoder = LabelEncoder()
#     encoder.fit(y_var)
#     y_var = encoder.transform(y_var)
    
#     # Resize images if required
#     if resize_images:
#         images = []
#         for img in x_var:
#             resized_img = skimage_resize(img, (128, 128, 3))
#             resized_img = 255 * resized_img  # Scale to 0-255 range
#             images.append(resized_img)
#         x_var = np.stack(images, axis=0)
        
#     else: # Resize to (56, 56, 3)
#         images = []
#         for img in x_var:
#             resized_img = skimage_resize(img, (56, 56, 3))
#             resized_img = 255 * resized_img  # Scale to 0-255 range
#             images.append(resized_img)
        
#         x_var = np.stack(images, axis=0)  # Corrected to stack resized images
        
#     # Create a DataFrame to hold the indices
#     indices = pd.DataFrame(range(len(x_var)), columns=['index'])

#     # Split the data into training and testing sets
#     x_train1, x_test1, y_train1, y_test1, index_train, index_test = train_test_split(
#         x_var, y_var, indices, train_size=0.8, random_state=RANDOM_STATE
#     )

#     return x_train1, x_test1, y_train1, y_test1, index_train, index_test


# # Function to update model name based on conditions
# def update_model_name(model_name):
#     if 'TRANSF' in model_name:
#         return 'ResNet-50'
#     elif 'SMALLCNN' in model_name:
#         return 'CNN'
#     elif 'KNN' in model_name:
#         return 'KNN'
#     elif 'RF' in model_name:
#         return 'RF'
#     else:
#         return 'Unknown'
    
# # Dictionary to store predictions and corresponding indices
# predictions = {}

# for model_name, model in models.items():
#     if model_name.endswith('.pkl'):
#         x_train, x_test, y_train, y_test, index_train, index_test = subset_split(stats_df, y)
#     elif 'smallcnn' in model_name.lower():
#         x_train, x_test, y_train, y_test, index_train, index_test = subset_split1(X, y, resize_images=False)
#     elif 'transf' in model_name.lower():
#         x_train, x_test, y_train, y_test, index_train, index_test = subset_split1(X, y, resize_images=True)
#     else:
#         continue  # Skip models that do not match the criteria

#     # Check if the model has a predict method
#     if hasattr(model, 'predict'):
#         print(f"Model {model_name} is a valid model object of type {type(model)}.")
        
#         # Generate predictions based on the model type
#         if isinstance(model, tf.keras.Model):
#             # Keras model
#             lr_probs = model.predict(x_test).ravel()  # Ensure the output is a 1D array
#         else:
#             # Scikit-learn model
#             lr_probs = model.predict(x_test)

#         # Convert probabilities to binary predictions
#         y_pred = (lr_probs > 0.5).astype(int)

#         # Store predictions and corresponding indices
#         predictions[model_name] = {
#             'y_pred': y_pred,
#             'index_test': index_test,
#             'y_test':y_test
#         }
#     else:
#         print(f"Model {model_name} is not recognized as a valid model object.")

# # Create a DataFrame to store the results
# results = []

# # Mapping for concdateID to flowering_date_uav_estimate
# concdateID_to_flowering_date = {
#     'M1111': 247,
#     'M0111': 262,
#     'M0011': 279
#     #'M0001': 296
# }

# # Iterate over the predictions to create the results DataFrame
# for model_name, data in predictions.items():
#     indices = data['index_test'].values  # Corrected to access the values directly
#     y_pred = data['y_pred']
#     y_test = data['y_test']  # Assuming y_test is available in the data dictionary
#     for i, idx in enumerate(indices):
#         concdate = concdateID.iloc[idx].item()
#         flowering_date = floweringdate.iloc[idx].item()
#         flowering_date_uav_estimate = concdateID_to_flowering_date.get(concdate, None)
#         results.append({
#             'pred_test': y_pred[i],
#             'y_test': y_test[i],  # Add y_test to the results
#             'model_name': model_name,
#             'index_test': idx,
#             'concdateID': concdate,
#             'flowering_date': flowering_date,
#             'flowering_date_uav_estimate': flowering_date_uav_estimate
#         })
# # Convert results to DataFrame
# results_df = pd.DataFrame(results)


# # Calculate metrics
# metrics = []
# for model_name, group in results_df.groupby('model_name'):
#     accuracy = accuracy_score(group['y_test'], group['pred_test'])
#     precision = precision_score(group['y_test'], group['pred_test'])
#     recall = recall_score(group['y_test'], group['pred_test'])
#     f1 = f1_score(group['y_test'], group['pred_test'])
    
#     metrics.append({
#         'model_name': model_name,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     })

# # Convert metrics to DataFrame
# metrics_df = pd.DataFrame(metrics)

# # Merge metrics with results DataFrame
# results_df = results_df.merge(metrics_df, on='model_name')

# # Print the results DataFrame
# print("Results DataFrame:")
# print(results_df)

# # Save the results DataFrame to a CSV file
# results_df.to_csv(os.path.join('../output/models_performance and figures', 'benchmark_models.csv'), index=False)

# print('CSV saved successfully.')


# # # Print the results DataFrame
# # print("Results DataFrame:")
# # print(results_df)
# # # Save the results DataFrame to a CSV file
# # #results_df.to_csv(os.path.join(dir_out,'all_benchmark_models.csv'), index=False)
# # results_df.to_csv(os.path.join('../output/models_performance and figures', 'benchmark_models.csv'), index=False)


# Load images and corresponding ground truth
from directories import dir_img, dir_img1, dir_gt, dir_out
from data_loader import load_dataset1, load_datasett, load_dataset, load_datasettt
#from other_utils import subset_split, subset_split1, store_roc_results, summary_metrics 
import os
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize as skimage_resize
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
X, y, floweringdate, concdateID = load_datasett(dir_img, dir_gt)

models_dir = '../output/bestmodels/benchmarks1'
models = {}

# Iterate through files in the models directory
for file_name in os.listdir(models_dir):
    file_path = os.path.join(models_dir, file_name)
    
    # Load .pkl models using joblib
    if file_name.endswith('.pkl'):
        models[file_name] = joblib.load(file_path)
    
    # Load .h5 models using Keras load_model
    elif file_name.endswith('.h5'):
        models[file_name] = load_model(file_path)

# Print the loaded models
print("Loaded models:")
for model_name, model in models.items():
    print(f"{model_name}: {model}")

def calculate_tabular_stats(X):
    statslist = []
    for p in X:
        # Calculate median for each channel
        bmean = np.median(p[:,:,0])
        gmean = np.median(p[:,:,1])
        rmean = np.median(p[:,:,2])
        
        # Calculate 99th percentile for each channel
        bperc99 = np.percentile(p[:,:,0], 99)
        gperc99 = np.percentile(p[:,:,1], 99)
        rperc99 = np.percentile(p[:,:,2], 99)
        
        # Calculate range (max - min) for each channel
        brange = np.ptp(p[:,:,0])
        grange = np.ptp(p[:,:,1])
        rrange = np.ptp(p[:,:,2])
        
        # Calculate standard deviation for each channel
        bstd = np.std(p[:,:,0])
        gstd = np.std(p[:,:,1])
        rstd = np.std(p[:,:,2])
        
        # Calculate max value for each channel
        bmax = np.max(p[:,:,0])
        gmax = np.max(p[:,:,1])
        rmax = np.max(p[:,:,2])
        
        # Create a row of statistics
        means_row = (bmean, gmean, rmean, bperc99, gperc99, rperc99, brange, grange, rrange, bstd, gstd, rstd, bmax, gmax, rmax)
        
        # Append the row to the stats list
        statslist.append(means_row)

    # Convert the list of statistics to a DataFrame
    stats_df = pd.DataFrame(np.stack(statslist))

    # Assign column names to the DataFrame
    stats_df.columns = ['Blue_med', 'Green_med', 'Red_med', 'Blue_p99', 'Green_p99', 'Red_perc99', 'Blue_range',
                        'Green_range', 'Red_range', 'bstd', 'gstd', 'rstd', 'bmax', 'gmax', 'rmax']

    return stats_df

# Call tabular stats generation
stats_df = calculate_tabular_stats(X)

print(stats_df.head())

def subset_split(x_var, y_var, RANDOM_STATE=123, n_classes=2):
    # Encode the labels
    encoder = LabelEncoder()
    encoder.fit(y_var)
    y_var = encoder.transform(y_var)

    # Create a DataFrame to hold the indices
    indices = pd.DataFrame(x_var.index)

    # Split the data into training and testing sets
    x_train1, x_test1, y_train1, y_test1, index_train, index_test = train_test_split(
        x_var, y_var, indices, train_size=0.8, random_state=123)

    return x_train1, x_test1, y_train1, y_test1, index_train, index_test

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

def subset_split1(x_var, y_var, RANDOM_STATE=123, n_classes=2, resize_images=False):
    # Encode the labels
    encoder = LabelEncoder()
    encoder.fit(y_var)
    y_var = encoder.transform(y_var)
    
    # Resize images if required
    if resize_images:
        images = []
        for img in x_var:
            
            print(f"Shape: {img.shape}")
            print(f"Min pixel value: {np.min(img)}")
            print(f"Max pixel value: {np.max(img)}")
            resized_img = skimage_resize(img, (128, 128, 3))
            # Print shape, min, and max pixel values of resized_img
            print(f"Shape: {resized_img.shape}")
            print(f"Min pixel value: {np.min(resized_img)}")
            print(f"Max pixel value: {np.max(resized_img)}")
            resized_img = 255 * resized_img  # Scale to 0-255 range
            print(f"Shape: {resized_img.shape}")
            print(f"Min pixel value: {np.min(resized_img)}")
            print(f"Max pixel value: {np.max(resized_img)}")
            resize_images = preprocess_input(resized_img)
            print(f"Shape: {resize_images.shape}")
            print(f"Min pixel value: {np.min(resize_images)}")
            print(f"Max pixel value: {np.max(resize_images)}")
            
            images.append(resize_images)
        x_var = np.stack(images, axis=0)
        
    else: # Resize to (56, 56, 3)
        images = []
        for img in x_var:
            resized_img = skimage_resize(img, (56, 56, 3))
            #resized_img = 255 * resized_img  # Scale to 0-255 range
            images.append(resized_img)
        
        x_var = np.stack(images, axis=0)  # Corrected to stack resized images
        
    # Create a DataFrame to hold the indices
    indices = pd.DataFrame(range(len(x_var)), columns=['index'])

    # Split the data into training and testing sets
    x_train1, x_test1, y_train1, y_test1, index_train, index_test = train_test_split(
        x_var, y_var, indices, train_size=0.8, random_state=123
    )

    return x_train1, x_test1, y_train1, y_test1, index_train, index_test

# Function to update model name based on conditions
def update_model_name(model_name):
    if 'TRANSF' in model_name:
        return 'ResNet-50'
    elif 'SMALLCNN' in model_name:
        return 'CNN'
    elif 'KNN' in model_name:
        return 'KNN'
    elif 'RF' in model_name:
        return 'RF'
    else:
        return 'Unknown'

# Dictionary to store predictions and corresponding indices
predictions = {}

for model_name, model in models.items():
    if model_name.endswith('.pkl'):
        x_train, x_test, y_train, y_test, index_train, index_test = subset_split(stats_df, y)
    elif 'smallcnn' in model_name.lower():
        x_train, x_test, y_train, y_test, index_train, index_test = subset_split1(X, y, resize_images=False)
    elif 'transf' in model_name.lower():
        x_train, x_test, y_train, y_test, index_train, index_test = subset_split1(X, y, resize_images=True)
    else:
        continue  # Skip models that do not match the criteria

    # Check if the model has a predict method
    if hasattr(model, 'predict'):
        print(f"Model {model_name} is a valid model object of type {type(model)}.")
        
        # Generate predictions based on the model type
        if isinstance(model, tf.keras.Model):
            # Keras model
            lr_probs = model.predict(x_test).ravel()  # Ensure the output is a 1D array
            y_pred = (lr_probs > 0.5).astype(int)  # Convert probabilities to binary predictions
        else:
            # Scikit-learn model
            lr_probs = model.predict(x_test)
            y_pred = (lr_probs > 0.5).astype(int) 

        # Store predictions and corresponding indices
        predictions[model_name] = {
            'y_pred': y_pred,
            'index_test': index_test,
            'y_test': y_test
        }
    else:
        print(f"Model {model_name} is not recognized as a valid model object.")

# Create a DataFrame to store the results
results = []

# Mapping for concdateID to flowering_date_uav_estimate
concdateID_to_flowering_date = {
    'M1111': 247,
    'M0111': 262,
    'M0011': 279
    #'M0001': 296
}


# Sample size label mapping
sample_size_mapping = {
    30: 1,
    60: 2,
    100: 3,
    300: 10,
    900: 30,
    1800: 60,
    2400: 80,
    3000: 100
}

for model_name, data in predictions.items():
    # Extract sample_size_label and updated model_name
    parts = model_name.split('_')
    sample_size_label = int(parts[0])
    updated_model_name = update_model_name(parts[1])
    
    indices = data['index_test'].values  # Corrected to access the values directly
    y_pred = data['y_pred']
    y_test = data['y_test']  # Assuming y_test is available in the data dictionary
    for i, idx in enumerate(indices):
        concdate = concdateID.iloc[idx].item()
        flowering_date = floweringdate.iloc[idx].item()
        flowering_date_uav_estimate = concdateID_to_flowering_date.get(concdate, None)
        sample_size_label1 = sample_size_mapping.get(sample_size_label, None)
        results.append({
            'pred_test': y_pred[i],
            'y_test': y_test[i],  # Add y_test to the results
            'model_name': updated_model_name,
            'model_name_original': model_name,
            'sample_size_label': sample_size_label,
            'sample_size_label1': sample_size_label1,
            'index_test': idx,
            'concdateID': concdate,
            'flowering_date': flowering_date,
            'flowering_date_uav_estimate': flowering_date_uav_estimate
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate metrics
# Calculate metrics
metrics = []
for model_name_original, group in results_df.groupby('model_name_original'):
    
    model_name = group['model_name_original'].iloc[0]
    updated_model_name = group['model_name'].iloc[0]
    sample_size_label = group['sample_size_label'].iloc[0]
    
    sample_size_label1 = sample_size_mapping.get(sample_size_label, None)
    
    accuracy = accuracy_score(group['y_test'], group['pred_test'])
    precision = precision_score(group['y_test'], group['pred_test'])
    recall = recall_score(group['y_test'], group['pred_test'])
    f1 = f1_score(group['y_test'], group['pred_test'])
    
    metrics.append({
        'model_name': updated_model_name,
        'model_name_original': model_name,
        'sample_size_label': sample_size_label,
        'sample_size_label1': sample_size_label1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Print the results DataFrame
print("Results DataFrame:")
print(metrics_df)

# Save the results DataFrame to a CSV file
metrics_df.to_csv(os.path.join('../output/models_performance and figures', 'benchmark_models_metrics.csv'), index=False)

print('CSV saved successfully.')