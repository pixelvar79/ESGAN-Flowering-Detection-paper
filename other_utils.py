import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from numpy.random import randint
import seaborn as sns, matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,roc_curve, auc,roc_auc_score,f1_score,jaccard_score,precision_score,recall_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.transform import resize as skimage_resize
import numpy as np
from numpy.random import randint

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
    stats_df.columns = ['Blue_med', 'Green_med', 'Red_med', 'Blue_p99', 'Green_p99', 'Red_p99', 'Blue_range',
                        'Green_range', 'Red_range', 'bstd', 'gstd', 'rstd', 'bmax', 'gmax', 'rmax']

    return stats_df


def subset_split(x_var, y_var, nsample, RANDOM_STATE=42, n_classes=2):
    
    x_train1 = pd.DataFrame()
    y_train1 = pd.DataFrame()
    
    # Encode the labels
    encoder = LabelEncoder()
    encoder.fit(y_var)
    y_var = encoder.transform(y_var)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, train_size=0.8, random_state=RANDOM_STATE)

    # balanced number of samples per class
    n_per_class = int(nsample / n_classes)
    for i in range(n_classes):
        # Get all samples for this class
        X_with_class = pd.DataFrame(x_train[y_train == i])
        Y_with_class = pd.DataFrame(y_train[y_train == i])
        
        # Choose random samples for each class
        ix = randint(0, len(X_with_class), n_per_class)
        x_train1 = x_train1.append(X_with_class.iloc[ix], ignore_index=True)
        y_train1 = y_train1.append(Y_with_class.iloc[ix], ignore_index=True)
    
    # Split the subset into training and validation sets
    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1, train_size=0.8, random_state=RANDOM_STATE)
    
    return x_train1, x_val, y_train1, y_val


def subset_split1(x_var, y_var, nsample, RANDOM_STATE=42, n_classes=2, resize_images=False):
    # Encode the labels
    encoder = LabelEncoder()
    encoder.fit(y_var)
    y_var = encoder.transform(y_var)
    
    # Resize images if required
    if resize_images:
        images = []
        for img in x_var:
            resized_img = skimage_resize(img, (128, 128, 3))
            resized_img = 255 * resized_img  # Scale to 0-255 range
            images.append(resized_img)
        x_var = np.stack(images, axis=0)
        
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, train_size=0.8, random_state=RANDOM_STATE)
    
    X_list, y_list = list(), list()
    n_per_class = int(nsample / n_classes)  # Number of samples per class
    
    for i in range(n_classes):
        X_with_class = x_train[y_train == i]  # Get all images for this class
        ix = randint(0, len(X_with_class), n_per_class)  # Choose random images for each class
        [X_list.append(X_with_class[j]) for j in ix]  # Add to list
        [y_list.append(i) for j in ix]
    
    x_train1 = np.asarray(X_list)
    y_train1 = np.asarray(y_list)
    
    # Split the subset into training and validation sets
    x_train1, x_val, y_train1, y_val = train_test_split(x_train1, y_train1, train_size=0.8, random_state=RANDOM_STATE)
    
    return x_train1, x_val, y_train1, y_val


import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

# def store_roc_results(model_name, iteration, x_test, y_test):
#     list_fpr = []
#     list_tpr = []
    
#     #saved_model = tf.keras.models.load_model(model_name)
#     # generate a no skill prediction (majority class)
#     ns_probs = [0 for _ in range(len(y_test))]
#     lr_probs = model_name.predict(x_test)

#     # calculate roc curves
#     ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
#     lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    
#     list_fpr.append(lr_fpr)
#     list_tpr.append(lr_tpr)
    
#     # Save the results to a local folder
#     results_dir = 'roc_results'
#     os.makedirs(results_dir, exist_ok=True)
#     np.save(os.path.join(results_dir, f'{model_name}_fpr_{iteration}.npy'), list_fpr)
#     np.save(os.path.join(results_dir, f'{model_name}_tpr_{iteration}.npy'), list_tpr)

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

def store_roc_results(model, nsample, iteration, x_test, y_test, outdir):
    list_fpr = []
    list_tpr = []
    
    # Extract the class name of the model
    model_name = model.__class__.__name__
    
    # Generate predictions based on the model type
    if isinstance(model, tf.keras.Model):
        # Keras model
        lr_probs = model.predict(x_test).ravel()  # Ensure the output is a 1D array
    else:
        # Scikit-learn model
        lr_probs = model.predict(x_test)# Get probabilities for the positive class
    
    # Generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # Calculate ROC curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    
    list_fpr.append(lr_fpr)
    list_tpr.append(lr_tpr)
    

    np.save(os.path.join(outdir, f'{model_name}_{nsample}_{iteration}_fpr.npy'), list_fpr)
    np.save(os.path.join(outdir, f'{model_name}_{nsample}_{iteration}_tpr.npy'), list_tpr)
    

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, jaccard_score, precision_score, recall_score


import pandas as pd

def summary_metrics(model, nsample, y_val, preds, outdir, iter):
    # Assuming model_name is a string representing the model's name
    model_name = model.__class__.__name__
    
    # Create a DataFrame to store the metrics
    mdf = pd.DataFrame()
    
    # Calculate metrics (example)
    mdf['accuracy'] = [accuracy_score(y_val, preds)]
    mdf['precision'] = [precision_score(y_val, preds, average='weighted')]
    mdf['recall'] = [recall_score(y_val, preds, average='weighted')]
    mdf['f1'] = [f1_score(y_val, preds, average='weighted')]
    
    # Add model name to the DataFrame
    mdf['model'] = model_name
    
    # Save the DataFrame to a CSV file
    mdf.to_csv(f'{outdir}/metrics_{model_name}_{nsample}_{iter}.csv', index=False)

# def summary_metrics(model_name, nsample, y_true, y_pred, iteration):
#     accur = []
#     roc = []
#     f1 = []
#     jaccard = []
#     precision = []
#     recall = []

#     m1 = accuracy_score(y_true, y_pred)
#     m2 = roc_auc_score(y_true, y_pred)
#     m3 = f1_score(y_true, y_pred)
#     m4 = jaccard_score(y_true, y_pred, average='binary')
#     m5 = precision_score(y_true, y_pred)
#     m6 = recall_score(y_true, y_pred)
#     accur.append(np.asarray(m1))
#     roc.append(np.asarray(m2))
#     f1.append(np.asarray(m3))
#     jaccard.append(np.asarray(m4))
#     precision.append(np.asarray(m5))
#     recall.append(np.asarray(m6))

#     mdf = pd.DataFrame()
#     mdf['accuracy'] = pd.DataFrame(accur)
#     mdf['roc_score'] = pd.DataFrame(roc)
#     mdf['f1'] = pd.DataFrame(f1)
#     mdf['jaccard'] = pd.DataFrame(jaccard)
#     mdf['precision'] = pd.DataFrame(precision)
#     mdf['recall'] = pd.DataFrame(recall)
#     mdf['label'] = nsample
#     mdf['model'] = model_name

#     # Save the results to a local folder
#     results_dir = 'summary_metrics'
#     os.makedirs(results_dir, exist_ok=True)
#     mdf.to_csv(os.path.join(results_dir, f'{model_name}_metrics_{iteration}.csv'), index=False)



