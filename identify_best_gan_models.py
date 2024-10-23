
from directories import dir_img, dir_img1, dir_gt, dir_out
from data_loader import load_dataset1, load_dataset, load_datasett

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob

import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import glob

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

# # Directory containing GAN models
# _, x_test, _, y_test, index_train, index_test = load_real_samples(X,y)

# # Print shapes of x_test and y_test
# print(f"x_test shape: {x_test.shape}")
# print(f"y_test shape: {y_test.shape}")

# Directory containing GAN models
dir_source = os.path.join('../output/head50/gan', 'c_model_*_GAN_*_0_111.h5')

# Dictionary to store predictions and corresponding indices
predictions = {}

# Load GAN models and generate predictions
model_paths = glob.glob(dir_source)#[:60]  # Limit to first 20 models

# print(model_paths)


for model_path in model_paths:
    model_name = os.path.basename(model_path)
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the data
    x_train, x_test, y_train, y_test, index_train, index_test = load_real_samples(X, y)
    
    
    # Check if the model has a predict method
    if hasattr(model, 'predict'):
        print(f"Model {model_name} is a valid model object of type {type(model)}.")
        
        # Generate predictions
        lr_probs = model.predict(x_test)  # Get the probabilities
        y_pred = (lr_probs[:, 1] > 0.5).astype(int).reshape(-1, 1).flatten()# Convert to binary predictions and reshape

        # # Print y_pred and y_test values
        # print(f"y_pred: {y_pred}")
        # print(f"y_test: {y_test.reshape(-1, 1)}")  # Reshape y_test for comparison

        # Store predictions and corresponding indices
        predictions[model_name] = {
            'y_pred': y_pred,
            'index_test': index_test,
            'y_test': y_test
        }
        #         # Print shapes of stored predictions
        # print(f"predictions[{model_name}] y_pred shape: {y_pred.shape}")
        # print(f"predictions[{model_name}] index_test shape: {index_test.shape}")
        # print(f"predictions[{model_name}] y_test shape: {y_test.shape}")
  
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

# Iterate over the predictions to create the results DataFrame
for model_name, data in predictions.items():
    indices = data['index_test'].values  # Corrected to access the values directly
    y_pred = data['y_pred']
    y_test = data['y_test']  # Assuming y_test is available in the data dictionary
    for i, idx in enumerate(indices):
        concdate = concdateID.iloc[idx].item()
        flowering_date = floweringdate.iloc[idx].item()
        flowering_date_uav_estimate = concdateID_to_flowering_date.get(concdate, None)
        results.append({
            'pred_test': y_pred[i],
            'y_test': y_test[i],  # Add y_test to the results
            'model_name': model_name,
            'index_test': idx,
            'concdateID': concdate,
            'flowering_date': flowering_date,
            'flowering_date_uav_estimate': flowering_date_uav_estimate
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)


# Extract models' labels from filename and calculate metrics
metrics = []
for model_name, data in predictions.items():
    y_val = data['y_test']
    preds = data['y_pred']
    
    # Extract labels from filename
    bases = Path(model_name).stem
    filename = bases.split('_')
    ssize_label = filename[-3]
    step_label = filename[-5]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    
    # Store metrics
    metrics.append({
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'e_step_label': step_label,
        'sample_size_label': ssize_label,
        'model_name': 'ESGAN'
        
    })

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Merge metrics with results DataFrame
results_df = results_df.merge(metrics_df, on='model_name')

# # Print the results DataFrame
print("Results DataFrame:")
print(results_df)

# Save the results DataFrame to a CSV file
#results_df.to_csv(os.path.join(dir_out, 'all_gan_models.csv'), index=False)
results_df.to_csv(os.path.join('../output/models_performance and figures', 'all_gan_models.csv'), index=False)


# Find the best models for each group of sample_size_label
best_models_idx = results_df.groupby('sample_size_label')['accuracy'].idxmax()

# Subset the DataFrame to keep only the best models
best_models_df = results_df.loc[best_models_idx]

# Print the subset DataFrame
print("Best Models DataFrame:")
print(best_models_df)

# Save the subset DataFrame to a CSV file
#best_models_df.to_csv(os.path.join(dir_out, 'best_gan_models.csv'), index=False)
best_models_df.to_csv(os.path.join('../output/models_performance and figures', 'best_gan_models.csv'), index=False)
