
# from directories import dir_img, dir_img1, dir_gt, dir_out
# from data_loader import load_dataset1, load_dataset, load_datasett
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# import os
# import tensorflow as tf
# import glob
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from pathlib import Path

# # Load dataset
# X, y, floweringdate, concdateID = load_datasett(dir_img, dir_gt)

# # Function to load and preprocess real samples
# def load_real_samples(X111, y111):
#     # Normalize the images to the range [-1, 1]
#     X1111 = (X111 - 0.5) / 0.5
#     # Encode the labels as integers
#     encoder = LabelEncoder()
#     encoder.fit(y111)
#     y1111 = encoder.transform(y111)
#     # Create a DataFrame to hold the indices
#     indices = pd.DataFrame(range(len(X1111)), columns=['index'])
#     # Split the data into training and testing sets
#     x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(
#         X1111, y1111, indices, train_size=0.8, random_state=123
#     )
#     return [x_train, x_test, y_train, y_test, index_train, index_test]

# # # Directory containing GAN models
# # _, x_test, _, y_test, index_train, index_test = load_real_samples(X,y)

# # # Print shapes of x_test and y_test
# # print(f"x_test shape: {x_test.shape}")
# # print(f"y_test shape: {y_test.shape}")

# # Directory containing GAN models
# # Directory containing GAN models
# dir_source = '../output/head50/gan'

# # List of specific model filenames
# # model_filenames = [
# #     'c_model_7800_GAN_100_0_111.h5',
# #     'c_model_0372_GAN_1800_0_111.h5',
# #     'c_model_5642_GAN_2400_0_111.h5',
# #     'c_model_12090_GAN_2800_0_111.h5',
# #     'c_model_5642_GAN_300_0_111.h5',
# #     'c_model_29232_GAN_32_0_111.h5',
# #     'c_model_40194_GAN_60_0_111.h5',
# #     'c_model_0434_GAN_900_0_111.h5'
# # ]

# model_filenames = [
#     'c_model_6760_GAN_100_0_111.h5',
#     'c_model_2666_GAN_1800_0_111.h5',
#     'c_model_3534_GAN_2400_0_111.h5',
#     'c_model_12152_GAN_2800_0_111.h5',
#     'c_model_8556_GAN_300_0_111.h5',
#     'c_model_28449_GAN_32_0_111.h5',
#     'c_model_35496_GAN_60_0_111.h5',
#     'c_model_0434_GAN_900_0_111.h5'
# ]

# # Dictionary to store predictions and corresponding indices
# predictions = {}

# # Load GAN models and generate predictions
# for model_filename in model_filenames:
#     model_path = os.path.join(dir_source, model_filename)
#     model_name = os.path.basename(model_path)
#     model = tf.keras.models.load_model(model_path)
    
#     # Load and preprocess the data
#     x_train, x_test, y_train, y_test, index_train, index_test = load_real_samples(X, y)
    
#     # Print shapes of x_test and y_test
#     print(f"x_test shape: {x_test.shape}")
#     print(f"y_test shape: {y_test.shape}")
    
    
#     # Check if the model has a predict method
#     if hasattr(model, 'predict'):
#         print(f"Model {model_name} is a valid model object of type {type(model)}.")
        
#         # Generate predictions
#         lr_probs = model.predict(x_test)  # Get the probabilities
#         y_pred = (lr_probs[:, 1] > 0.5).astype(int).reshape(-1, 1).flatten()# Convert to binary predictions and reshape

#         # # Print y_pred and y_test values
#         # print(f"y_pred: {y_pred}")
#         # print(f"y_test: {y_test.reshape(-1, 1)}")  # Reshape y_test for comparison

#         # Store predictions and corresponding indices
#         predictions[model_name] = {
#             'y_pred': y_pred,
#             'index_test': index_test,
#             'y_test': y_test
#         }
#         #         # Print shapes of stored predictions
#         # print(f"predictions[{model_name}] y_pred shape: {y_pred.shape}")
#         # print(f"predictions[{model_name}] index_test shape: {index_test.shape}")
#         # print(f"predictions[{model_name}] y_test shape: {y_test.shape}")
  
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


# # Extract models' labels from filename and calculate metrics
# metrics = []
# for model_name, data in predictions.items():
#     y_val = data['y_test']
#     preds = data['y_pred']
    
#     # Extract labels from filename
#     bases = Path(model_name).stem
#     filename = bases.split('_')
#     ssize_label = filename[-3]
#     step_label = filename[-5]
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_val, preds)
#     precision = precision_score(y_val, preds)
#     recall = recall_score(y_val, preds)
#     f1 = f1_score(y_val, preds)
    
#     # Store metrics
#     metrics.append({
#         'model_name': model_name,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'e_step_label': step_label,
#         'sample_size_label': ssize_label
#     })

# # Convert metrics to DataFrame
# metrics_df = pd.DataFrame(metrics)

# # Merge metrics with results DataFrame
# results_df = results_df.merge(metrics_df, on='model_name')

# # # Print the results DataFrame
# print("Results DataFrame:")
# print(results_df)

# # Save the results DataFrame to a CSV file
# results_df.to_csv(os.path.join(dir_out, 'flowering_determination_best_gan_models.csv'), index=False)

# # Find the best models for each group of sample_size_label
# best_models_idx = results_df.groupby('sample_size_label')['accuracy'].idxmax()

# # Subset the DataFrame to keep only the best models
# best_models_df = results_df.loc[best_models_idx]

# # Print the subset DataFrame
# print("Best Models DataFrame:")
# print(best_models_df)

# # Save the subset DataFrame to a CSV file
# best_models_df.to_csv(os.path.join(dir_out, 'best_flowering_determination_best_gan_models.csv'), index=False)


from directories import dir_img, dir_img1, dir_gt, dir_out
from data_loader import load_dataset1, load_dataset, load_datasett
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import tensorflow as tf
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Load dataset
X, y, floweringdate, concdateID = load_datasett(dir_img, dir_gt)

# Function to load and preprocess real samples
def load_real_samples(X111, y111):
    # Normalize the images to the range [-1, 1]
    X1111 = (X111 - 0.5) / 0.5
    # Encode the labels as integers
    encoder = LabelEncoder()
    encoder.fit(y111)
    y1111 = encoder.transform(y111)
    # Create a DataFrame to hold the indices
    indices = pd.DataFrame(range(len(X1111)), columns=['index'])
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(
        X1111, y1111, indices, train_size=0.8, random_state=123
    )
    return [x_train, x_test, y_train, y_test, index_train, index_test]

# Directory containing GAN models
dir_source = '../output/head50/gan'

# List of specific model filenames
model_filenames = [
    'c_model_6760_GAN_100_0_111.h5',
    'c_model_2666_GAN_1800_0_111.h5',
    'c_model_3534_GAN_2400_0_111.h5',
    'c_model_12152_GAN_2800_0_111.h5',
    'c_model_8556_GAN_300_0_111.h5',
    'c_model_28449_GAN_32_0_111.h5',
    'c_model_35496_GAN_60_0_111.h5',
    'c_model_0434_GAN_900_0_111.h5'
]

# Dictionary to store predictions and corresponding indices
predictions = {}




# Load GAN models and generate predictions
for model_filename in model_filenames:
    model_path = os.path.join(dir_source, model_filename)
    model_name = os.path.basename(model_path)
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the data
    x_train, x_test, y_train, y_test, index_train, index_test = load_real_samples(X, y)
    
    # Print shapes of x_test and y_test
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Check if the model has a predict method
    if hasattr(model, 'predict'):
        print(f"Model {model_name} is a valid model object of type {type(model)}.")
        
        # Generate predictions
        lr_probs = model.predict(x_test)  # Get the probabilities
        y_pred = (lr_probs[:, 1] > 0.5).astype(int).reshape(-1, 1).flatten()  # Convert to binary predictions and reshape

        # Store predictions and corresponding indices
        predictions[model_name] = {
            'y_pred': y_pred,
            'index_test': index_test,
            'y_test': y_test
        }
    else:
        print(f"Model {model_name} is not recognized as a valid model object.")
        
#print(predictions)

# Create a DataFrame to store the results
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

# print(results_df)

# # Extract models' labels from filename and calculate metrics
# metrics = []
# for model_name, data in predictions.items():
#     y_val = data['y_test']
#     preds = data['y_pred']
    
#     # Extract labels from filename
#     bases = Path(model_name).stem
#     filename = bases.split('_')
#     ssize_label = filename[-3]
#     step_label = filename[-5]
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_val, preds)
#     precision = precision_score(y_val, preds)
#     recall = recall_score(y_val, preds)
#     f1 = f1_score(y_val, preds)
    
#     # Store metrics
#     metrics.append({
#         'model_name': 'ESGAN',  # Set model_name to ESGAN
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'e_step_label': step_label,
#         'sample_size_label': ssize_label
#     })

# # Convert metrics to DataFrame
# metrics_df = pd.DataFrame(metrics)

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Assuming predictions, concdateID, floweringdate, and concdateID_to_flowering_date are defined
# Initialize results list
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
    32: 1,
    60: 2,
    100: 3,
    300: 10,
    900: 30,
    1800: 60,
    2400: 80,
    2800: 100
}

# Iterate over the predictions to create the results DataFrame
for model_name, data in predictions.items():
    indices = data['index_test'].values  # Ensure indices are flattened
    y_pred = data['y_pred']
    y_test = data['y_test']
    
    for i, idx in enumerate(indices):
        concdate = concdateID.iloc[idx].item()
        flowering_date = floweringdate.iloc[idx].item()
        flowering_date_uav_estimate = concdateID_to_flowering_date.get(concdate, None)
        #sample_size_label = data['sample_size_label'][i]  # Assuming sample_size_label is available in the data dictionary
        results.append({
            'pred_test': y_pred[i],
            'y_test': y_test[i],
            'model_name': model_name,
            #'sample_size_label': sample_size_label,
            'index_test': idx,
            'concdateID': concdate,
            'flowering_date': flowering_date,
            'flowering_date_uav_estimate': flowering_date_uav_estimate
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print(results_df)

# Extract models' labels from filename and calculate metrics
metrics = []
for model_name, data in predictions.items():
    y_val = data['y_test']
    preds = data['y_pred']
    
    # Extract labels from filename
    bases = Path(model_name).stem
    filename = bases.split('_')
    ssize_label = int(filename[-3])
    step_label = filename[-5]
    
    # Map sample_size_label to sample_size_label1
    sample_size_label1 = sample_size_mapping.get(ssize_label, None)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    
    metrics.append({
        'model_name': 'ESGAN',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sample_size_label': ssize_label,
        'sample_size_label1': sample_size_label1,
        'step_label': step_label
    })

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Display the final DataFrame
print(metrics_df)

# Save the results DataFrame to a CSV file
metrics_df.to_csv(os.path.join('../output/models_performance and figures', 'best_gan_models_metrics.csv'), index=False)

# # Find the best models for each group of sample_size_label


