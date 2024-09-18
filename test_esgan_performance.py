import gc
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from keras.models import load_model
from data_loader import load_real_samples  # Assuming this function is in data_loader.py

# Load test data
_, x_test, _, y_test = load_real_samples()

# Define directory source
dir_out = 'path_to_output_directory'  # Replace with the actual path
dir_source = dir_out + '/c_model_*_ESGAN_*_*.h5'

def model_performance(dir_files):
    gc.collect()
    
    for f in glob.glob(dir_files):
        # Extract models' labels from filename
        bases = Path(f).stem
        filename = bases.split('_')
        iteration_label = filename[-1]
        ssize_label = filename[-2]
        step_label = filename[-4]

        # Load each h5 model & perform predict
        saved_model = load_model(f)
        pred = saved_model.predict(x_test)
        pred = np.argmax(pred, axis=-1)
        
        # Prepare metrics
        accur = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average='weighted')
        recall = recall_score(y_test, pred, average='weighted')
        f1 = f1_score(y_test, pred, average='weighted')

        mdf = pd.DataFrame({
            'accuracy': [accur],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'e_step_label': [step_label],
            'sample_size_label': [ssize_label],
            'iteration_label': [iteration_label],
            'model': ['ESGAN']
        })
        
        # Save the DataFrame to a CSV file
        mdf.to_csv(os.path.join(dir_out, f'metrics_ESGAN_{ssize_label}_{step_label}_{iteration_label}.csv'), index=False)
        
        # Calculate ROC AUC scores and ROC curves
        ns_probs = [0 for _ in range(len(y_test))]
        lr_probs = saved_model.predict(x_test)
        lr_probs = lr_probs[:, 1]
        
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        
        # Save ROC results
        np.save(os.path.join(dir_out, f'ESGAN_{ssize_label}_{step_label}_{iteration_label}_fpr.npy'), lr_fpr)
        np.save(os.path.join(dir_out, f'ESGAN_{ssize_label}_{step_label}_{iteration_label}_tpr.npy'), lr_tpr)

# Evaluate model performance
model_performance(dir_source)