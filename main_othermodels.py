import os
import gc
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from numpy import ones
from PIL import Image as Img
import tensorflow as tf
import joblib

from directories import dir_img, dir_gt, dir_out
from data_loader import load_dataset
from models import train_knn, train_rf, train_smallcnn, train_transf
from other_utils import calculate_tabular_stats, subset_split, subset_split1, store_roc_results, summary_metrics  # Import the functions
from sklearn.preprocessing import binarize


# Load images and corresponding ground truth
X, y = load_dataset(dir_img, dir_gt)

# Call tabular stats generation
stats_df = calculate_tabular_stats(X)

# Display the first few rows of the DataFrame
print(stats_df.head())

# Initialize lists for storing results
dfs = [] 
list_fpr = []
list_tpr = []

percents = (30, 60, 100, 300, 900, 1800, 2400, 3000)

# Flag to indicate if train_transf is being used
is_train_transf = False

#for i in range(3):
for i in range(5):
    
    for nsample in percents:
        
        print(f'Iteration {i}') # Print the iteration number
        x_train1, x_val, y_train1, y_val, x_test, y_test = subset_split(stats_df, y, nsample)
        
        # Train KNN model
        knn_model = train_knn(x_train1, y_train1, nsample, dir_out, i)
        knn_preds = knn_model.predict(x_val)
        store_roc_results(knn_model, nsample, i, x_test, y_test, dir_out)  # Save ROC results for KNN
        knn_preds_binary = binarize(knn_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        summary_metrics(knn_model, nsample, y_test, knn_preds_binary, dir_out, i)  # Save summary metrics for KNN

        # Train RF model
        rf_model = train_rf(x_train1, y_train1, nsample, dir_out, i)
        rf_preds = rf_model.predict(x_val)
        store_roc_results(rf_model, nsample, i, x_test, y_test, dir_out)  # Save ROC results for RF
        rf_preds_binary = binarize(rf_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        summary_metrics(rf_model, nsample, y_test, rf_preds_binary, dir_out, i)  # Save summary metrics for RF

        x_train1, x_val, y_train1, y_val, x_test, y_test = subset_split1(X, y, nsample, resize_images=False)

        # Train SmallCNN model
        smallcnn_model = train_smallcnn(x_train1, y_train1, x_val, y_val, nsample, dir_out, i)
        smallcnn_preds = smallcnn_model.predict(x_val)
        store_roc_results(smallcnn_model,nsample, i, x_test, y_test, dir_out)  # Save ROC results for SmallCNN
        smallcnn_preds_binary = binarize(smallcnn_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        summary_metrics(smallcnn_model, nsample, y_test, smallcnn_preds_binary, dir_out, i)  # Save summary metrics for SmallCNN

        # Set the flag before calling train_transf
        x_train1, x_val, y_train1, y_val, x_test, y_test = subset_split1(X, y, nsample, resize_images=True)
        transf_model = train_transf(x_train1, y_train1, x_val, y_val, nsample, dir_out, i)
        transf_preds = transf_model.predict(x_val)
        store_roc_results(transf_model, nsample,  i, x_test, y_test, dir_out)  # Save ROC results for Transformer
        transf_preds_binary = binarize(transf_preds.reshape(-1, 1), threshold=0.5).ravel()  # Convert to binary class labels
        summary_metrics(transf_model, nsample, y_test, transf_preds_binary, dir_out, i)  # Save summary metrics for Transformer
        # Reset the flag after calling train_transf
        is_train_transf = False

