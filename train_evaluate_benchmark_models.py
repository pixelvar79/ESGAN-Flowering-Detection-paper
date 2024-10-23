
import os
import time
import pandas as pd
from sklearn.preprocessing import binarize
from directories import dir_img, dir_gt, dir_out, dir_img1
from data_loader import load_datasett, load_datasettt
from models import train_knn, train_rf, train_smallcnn, train_transf
from benchmarks_models_utils import save_predictions, calculate_tabular_stats, subset_split, subset_split1, store_roc_results, summary_metrics, save_predictions  # Import the functions


# Load images and corresponding ground truth for training each model
X, y, floweringdate, concdateID, grouping = load_datasett(dir_img, dir_gt)

# load images and corresponding ground truth for testing each model at each grouping
X1, y1, floweringdate1, concdateID1, grouping1 = load_datasettt(dir_img1, dir_gt)

# Call tabular stats generation
stats_df = calculate_tabular_stats(X)
stats_df1 = calculate_tabular_stats(X1)


# Display the first few rows of the DataFrame
print(stats_df.head())

# Initialize lists for storing results
dfs = [] 
list_fpr = []
list_tpr = []

#define percent of training data to use
percents = (30, 60, 100, 300, 900, 1800, 2400, 3000)

# Flag to indicate if train_transf is being used
is_train_transf = False

# Initialize dictionaries to store predictions and training times
predictions = {}
training_times = {}

#store results
all_results_df = []
all_metrics_df = []

#for i in range(3):
for i in range(1):
    
    for nsample in percents:
        
        print(f'Iteration {i}') # Print the iteration number
        x_train1, x_val, y_train1, y_val, x_test, y_test, index_train, index_test = subset_split(stats_df, y, nsample)
        
        _, _, _, _, x_test1, y_test1, index_train1, index_test1 = subset_split(stats_df1, y1, nsample)
        
        # Train KNN model
        start_time = time.time()  # Start timer
        knn_model = train_knn(x_train1, y_train1, nsample, dir_out, i)
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        training_times[f'{nsample}_KNN'] = elapsed_time  # Store elapsed time
    
        knn_preds = knn_model.predict(x_test1)
        store_roc_results(knn_model, nsample, i, x_test1, y_test1, dir_out)  # Save ROC results for KNN
        knn_preds_binary = binarize(knn_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        
        # Store predictions and corresponding indices
        predictions[f'{nsample}_KNN'] = {
            'y_pred': knn_preds_binary,
            'y_test': y_test1,
            'index_test': index_test1,
            'iteration': i
        }
        
        # Train RF model
        start_time = time.time()  # Start timer
        rf_model = train_rf(x_train1, y_train1, nsample, dir_out, i)
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        training_times[f'{nsample}_RF'] = elapsed_time  # Store elapsed time
    
        rf_preds = rf_model.predict(x_test1)
        store_roc_results(rf_model, nsample, i, x_test1, y_test1, dir_out)  # Save ROC results for RF
        rf_preds_binary = binarize(rf_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        
        # Store predictions and corresponding indices
        predictions[f'{nsample}_RF'] = {
            'y_pred': rf_preds_binary,
            'y_test': y_test1,
            'index_test': index_test1,
            'iteration': i
        }

        x_train1, x_val, y_train1, y_val, x_test, y_test, index_train, index_test = subset_split1(X, y, nsample, resize_images=False)
        _, _, _, _, x_test1, y_test1, index_train1, index_test1 = subset_split1(X1, y1, nsample, resize_images=False)
        
        # Train SmallCNN model
        start_time = time.time()  # Start timer
        smallcnn_model = train_smallcnn(x_train1, y_train1, x_val, y_val, nsample, dir_out, i)
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        training_times[f'{nsample}_SMALLCNN'] = elapsed_time  # Store elapsed time
        
        smallcnn_preds = smallcnn_model.predict(x_test1)
        store_roc_results(smallcnn_model, nsample, i, x_test1, y_test1, dir_out)  # Save ROC results for SmallCNN
        smallcnn_preds_binary = binarize(smallcnn_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        
        # Store predictions and corresponding indices
        predictions[f'{nsample}_SMALLCNN'] = {
            'y_pred': smallcnn_preds_binary,
            'y_test': y_test1,
            'index_test': index_test1,
            'iteration': i
        }

        # Set the flag before calling train_transf
        x_train1, x_val, y_train1, y_val, x_test, y_test, index_train, index_test = subset_split1(X, y, nsample, resize_images=True)
        _, _, _, _, x_test1, y_test1, index_train1, index_test1 = subset_split1(X1, y1, nsample, resize_images=True)
        
        # Train Transformer model
        start_time = time.time()  # Start timer
        transf_model = train_transf(x_train1, y_train1, x_val, y_val, nsample, dir_out, i)
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        training_times[f'{nsample}_TRANSF'] = elapsed_time  # Store elapsed time
        
        transf_preds = transf_model.predict(x_test1)
        store_roc_results(transf_model, nsample, i, x_test1, y_test1, dir_out)  # Save ROC results for Transformer
        transf_preds_binary = binarize(transf_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        # Store predictions and corresponding indices
        predictions[f'{nsample}_TRANSF'] = {
            'y_pred': transf_preds_binary,
            'y_test': y_test1,
            'index_test': index_test1,
            'iteration': i
        }

        # Reset the flag after calling train_transf
        is_train_transf = False
        
    # Save predictions and metrics for the current iteration
    results_df = save_predictions(predictions, concdateID1, floweringdate1, grouping1)
    metrics_df = summary_metrics(predictions)
    
    # Append the results to the lists
    all_results_df.append(results_df)
    all_metrics_df.append(metrics_df)

# Concatenate all iterations' results into single DataFrames
final_results_df = pd.concat(all_results_df, ignore_index=True)
final_metrics_df = pd.concat(all_metrics_df, ignore_index=True)

# Save the final DataFrames to CSV files
final_results_df.to_csv(os.path.join(dir_out, 'benchmarks_models_predictions_and_y_test.csv'), index=False)
final_metrics_df.to_csv(os.path.join(dir_out, 'benchmarks_models_metrics.csv'), index=False)

# Convert the training_times dictionary to a DataFrame
training_times_df = pd.DataFrame(list(training_times.items()), columns=['Model', 'Training Time'])

# Save the DataFrame to a CSV file
training_times_df.to_csv(os.path.join(dir_out, 'training_times.csv'), index=False)

print('All iterations results and metrics saved successfully.')
print('Training times saved successfully.')

