
# best_models_df.to_csv(os.path.join(dir_out, 'best_flowering_determination_best_gan_models.csv'), index=False)
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
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
from directories import dir_img, dir_img1, dir_gt, dir_out

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
    parts = model_name.split('_')
    sample_size_label = int(parts[-3])
    updated_model_name = 'ESGAN'
    
    indices = data['index_test'].values  # Corrected to access the values directly
    y_pred = data['y_pred']
    y_test = data['y_test']
    iteration_n = data['iteration']# Assuming y_test is available in the data dictionary
    for i, idx in enumerate(indices):
        concdate = concdateID.iloc[idx].item()
        flowering_date = floweringdate.iloc[idx].item()
        flowering_date_uav_estimate = concdateID_to_flowering_date.get(concdate, None)
        sample_size_label1 = sample_size_mapping.get(sample_size_label, None)
        
        results.append({
            'pred_test': y_pred[i],
            'y_test': y_test[i],  # Add y_test to the results
            'model_name': 'ESGAN',
            'model_name_original': 'ESGAN',
            'sample_size_label': sample_size_label,
            'sample_size_label1': sample_size_label1,
            'index_test': idx,
            'concdateID': concdate,
            'flowering_date': flowering_date,
            'flowering_date_uav_estimate': flowering_date_uav_estimate,
            'iteration_n': iteration_n
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(dir_out, 'gan_models_predictions_and_y_test.csv'), index=False)

print('Predictions and y_test values saved successfully.')

# Extract models' labels from filename and calculate metrics
metrics = []
for model_name, data in predictions.items():
        # Extract sample_size_label and updated model_name
        parts = model_name_original.split('_')
        sample_size_label = int(parts[0])
        updated_model_name = 'ESGAN'
        
        indices = data['index_test']
        y_pred = data['y_pred']
        y_test = data['y_test']
        iteration_n = data['iteration']
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        sample_size_label1 = sample_size_mapping.get(sample_size_label, None)
        
        metrics.append({
            'model_name': 'ESGAN'
            'model_name_original': 'ESGAN',
            'sample_size_label': sample_size_label,
            'sample_size_label1': sample_size_label1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iteration_n': iteration_n
        
        })

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Display the final DataFrame
print(metrics_df)

# Save the results DataFrame to a CSV file
metrics_df.to_csv(os.path.join(dir_out, 'gan_models_metrics.csv'), index=False)

# # Find the best models for each group of sample_size_label


