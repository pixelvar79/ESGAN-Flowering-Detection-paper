


# from data_loader import load_dataset1, load_dataset
# from directories import dir_img, dir_img1, dir_gt, dir_out
# import numpy as np
# import matplotlib.pyplot as plt

# # Load dataset
# X, df, floweringdate, concdateID = load_dataset1(dir_img1, dir_gt)

# # Extract relevant columns
# y = df['f50_head_0904']
# y1 = df['f50_head_0919']
# y11 = df['f50_head_1005']
# y111 = df['f50_head_1022']

# # Define dates
# dates = ['247', '262', '279', '296']
# # Count flowering events for each date
# flowering_counts = {
#     '247': (y == 1).sum(),
#     '262': (y1 == 1).sum(),
#     '279': (y11 == 1).sum(),
#     '296': (y111 == 1).sum()
# }

# # Count not-flowering events for each date
# not_flowering_counts = {
#     '247': (y == 0).sum(),
#     '262': (y1 == 0).sum(),
#     '279': (y11 == 0).sum(),
#     '296': (y111 == 0).sum()
# }

# from scipy.interpolate import PchipInterpolator
# import os

# total_counts = {date: flowering_counts[date] + not_flowering_counts[date] for date in dates}

# # Convert counts to percentages
# flowering_percentages = {date: (flowering_counts[date] / total_counts[date]) * 100 for date in dates}

# # Plot the percentages
# plt.figure(figsize=(10, 6))
# bar_width = 0.35
# index = np.arange(len(dates))

# # Plot flowering events (bottom part of the bar)
# plt.bar(index, list(flowering_percentages.values()), bar_width, color='orange', edgecolor='black', label='Flowered')

# # Add a smoother trend line using PCHIP interpolation
# x_new = np.linspace(index.min(), index.max(), 300)
# pchip = PchipInterpolator(index, list(flowering_percentages.values()))
# y_smooth = pchip(x_new)
# plt.plot(x_new, y_smooth, "r--", label='_nolegend_', color='black')

# plt.xlabel('Dates of field evaluation for determining flowering date (Julian Dates)', fontsize=14)
# plt.ylabel('Flowered Events (% of total)', fontsize=14)
# plt.xticks(index, dates, fontsize=12)
# plt.yticks([0, 20, 40, 60, 80, 100], fontsize=12)  # Manually set y-ticks
# plt.ylim(0, 110)  # Extend y-axis to 110
# plt.tight_layout()
# plt.savefig(os.path.join('../output/models_performance and figures', 'flowering_events_by_date.png'))
# plt.show()

# import pandas as pd
# import os

# # Define the path to the directory containing the CSV files
# directory_path = '../output/head501'

# # Define the specific CSV files to open
# csv_files = [
#     'modified_flowering_determination_best_gan_models.csv',
#     'modified_flowering_determination_benchmark_models.csv'
# ]

# # Construct the full file paths
# csv_file_paths = [os.path.join(directory_path, file) for file in csv_files]

# # Read the specific CSV files into DataFrames
# dataframes = [pd.read_csv(file) for file in csv_file_paths]

# # Optionally, you can print the names of the files and the first few rows of each DataFrame
# for file, df in zip(csv_file_paths, dataframes):
#     print(f'File: {file}')
#     print(df.tail())
#     print('\n')

# import pandas as pd
# import os

# # Define the path to the directory containing the CSV file
# directory_path = '../output/head501'

# # Define the specific CSV file to open
# csv_file = 'flowering_determination_benchmark_models.csv'

# # Construct the full file path
# csv_file_path = os.path.join(directory_path, csv_file)

# # Read the CSV file into a DataFrame
# df = pd.read_csv(csv_file_path)

# # Add a new column 'sample_size_label' by extracting the first string as a number from 'model_name'
# df['sample_size_label'] = df['model_name'].str.extract(r'(\d+)').astype(int)

# # Define the path to save the modified CSV file
# output_directory = '../output/head501'
# os.makedirs(output_directory, exist_ok=True)
# output_file_path = os.path.join(output_directory, csv_file)

# # Save the modified DataFrame to a new CSV file
# df.to_csv(output_file_path, index=False)

# print(f'Modified CSV file saved to: {output_file_path}')


# import pandas as pd
# import os

# # Define the path to the directory containing the CSV files
# directory_path = '../output/head501'

# # Define the specific CSV files to open
# csv_files = [
#     'flowering_determination_best_gan_models.csv',
#     'flowering_determination_benchmark_models.csv'
# ]

# # Construct the full file paths
# csv_file_paths = [os.path.join(directory_path, file) for file in csv_files]

# # Function to determine the final model name based on the pattern in model_name
# def determine_final_model_name(model_name):
#     if 'TRANSF' in model_name:
#         return 'ResNet-50'
#     elif 'GAN' in model_name:
#         return 'ESGAN'
#     elif 'SMALLCNN' in model_name:
#         return 'CNN'
#     elif 'KNN' in model_name:
#         return 'KNN'
#     elif 'RF' in model_name:
#         return 'RF'
#     else:
#         return 'Unknown'

# # Read the specific CSV files into DataFrames and add the new column
# for file_path in csv_file_paths:
#     df = pd.read_csv(file_path)
#     df['final_model_name'] = df['model_name'].apply(determine_final_model_name)
    
#     # Save the modified DataFrame to a new CSV file
#     output_file_path = os.path.join(directory_path, 'modified_' + os.path.basename(file_path))
#     df.to_csv(output_file_path, index=False)
#     print(f'Modified CSV file saved to: {output_file_path}')


# import pandas as pd
# import os

# # Define the path to the directory containing the CSV files
# directory_path = '../output/head501'

# # Define the specific CSV files to open
# csv_files = [
#     'modified_flowering_determination_best_gan_models.csv',
#     'modified_flowering_determination_benchmark_models.csv'
# ]

# # Construct the full file paths
# csv_file_paths = [os.path.join(directory_path, file) for file in csv_files]

# # Columns to subset
# columns_to_subset = ['final_model_name', 'flowering_date_uav_estimate', 'sample_size_label', 'pred_test', 'y_test']

# # Read the specific CSV files into DataFrames and subset the required columns
# dataframes = [pd.read_csv(file)[columns_to_subset] for file in csv_file_paths]

# # Concatenate the DataFrames into one
# combined_df = pd.concat(dataframes, ignore_index=True)

# print(combined_df.head())

# # Define the path to save the combined CSV file
# output_file_path = os.path.join(directory_path, 'combined_flowering_determination_models.csv')

# # Save the combined DataFrame to a new CSV file
# combined_df.to_csv(output_file_path, index=False)

# print(f'Combined CSV file saved to: {output_file_path}')

# import pandas as pd
# import os

# # Define the path to the combined CSV file
# file_path = '../output/head501/combined_flowering_determination_models.csv'

# # Read the CSV file into a DataFrame
# df = pd.read_csv(file_path)

# # Print unique values in the 'sample_size_label' column
# unique_sample_size_labels = df['sample_size_label'].unique()
# print('Unique sample_size_label values:', unique_sample_size_labels)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, jaccard_score, precision_score, recall_score
# import os

# # Define the path to the input CSV file
# file_path = '../output/head501/updated_combined_flowering_determination_models.csv'

# # Read the CSV file into a DataFrame
# df = pd.read_csv(file_path)

# # Initialize a list to store the results
# results = []

# # Group by the specified columns
# grouped = df.groupby(['final_model_name', 'sample_size_label1', 'flowering_date_uav_estimate'])

# # Calculate metrics for each group
# for name, group in grouped:
#     y_true = group['y_test']
#     y_pred = group['pred_test']
    
#     accuracy = accuracy_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_pred) if len(y_true.unique()) > 1 else float('nan')  # Handle case with only one class
#     f1 = f1_score(y_true, y_pred)
#     jaccard = jaccard_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
    
#     results.append({
#         'final_model_name': name[0],
#         'sample_size_label1': name[1],
#         'flowering_date_uav_estimate': name[2],
#         'accuracy': accuracy,
#         'roc_auc': roc_auc,
#         'f1_score': f1,
#         'jaccard_index': jaccard,
#         'precision': precision,
#         'recall': recall
#     })

# # Create a new DataFrame from the results
# metrics_df = pd.DataFrame(results)

# # Define the categories and order
# categories = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']

# # Split the DataFrame by 'flowering_date_uav_estimate'
# unique_dates = metrics_df['flowering_date_uav_estimate'].unique()

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# # Function to plot metrics
# def plot_metrics(df, date):
#     metrics = ['accuracy', 'roc_auc', 'f1_score', 'jaccard_index', 'precision', 'recall']
#     n_metrics = len(metrics)
    
#     fig, axes = plt.subplots(3, 2, figsize=(28, 30))  # 3 rows, 2 columns, overall size 28x30
#     sns.set(font_scale=1.9)
#     sns.set_style("ticks")
    
#     for ax, col in zip(axes.flatten(), metrics):
#         sns.barplot(data=df,
#                     x='sample_size_label1',
#                     y=col,
#                     hue='final_model_name',
#                     alpha=0.8,
#                     linewidth=2,
#                     palette=palette,
#                     order=sorted(df['sample_size_label1'].unique()),
#                     ax=ax)  # Order by sample_size_label1
        
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#         ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
#         sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#         ax.set_xlabel("% Annotated samples")
#         ax.set_ylim([0, 1])
#         ax.set_title(f'{col} for Flowering Date UAV Estimate {date}')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join('../output/models_performance and figures/', f'metrics_flowering_date_{date}.png'), bbox_inches='tight')
#     plt.close()

# # Loop through each unique date and plot the metrics
# for date in unique_dates:
#     date_df = metrics_df[metrics_df['flowering_date_uav_estimate'] == date]
#     date_df = date_df[date_df['final_model_name'].isin(categories)]
#     date_df['final_model_name'] = pd.Categorical(date_df['final_model_name'], categories=categories)
#     date_df = date_df.sort_values(by=['final_model_name', 'sample_size_label1'], ascending=[True, True])
    
#     plot_metrics(date_df, date)

# print('Plots saved successfully.')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure the output directory exists
dir_fig = '../output/models_performance and figures'
os.makedirs(dir_fig, exist_ok=True)

# Load the CSV files into DataFrames
benchmark_df = pd.read_csv(os.path.join(dir_fig,'benchmark_models_metrics.csv'))
best_gan_df = pd.read_csv(os.path.join(dir_fig,'best_gan_models_metrics.csv'))

# Define the columns to keep
columns_to_keep = ['model_name', 'sample_size_label1', 'accuracy', 'precision', 'recall', 'f1']

# Subset the columns needed from each DataFrame
benchmark_df = benchmark_df[columns_to_keep]
best_gan_df = best_gan_df[columns_to_keep]

# Combine the DataFrames
both = pd.concat([benchmark_df, best_gan_df], ignore_index=True)

# Define a custom palette with hexadecimal color codes
palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

plt.figure(figsize=(14, 10))
sns.set(font_scale=1.9)
sns.set_style("ticks")


def plot_metrics(col):
    plt.figure()
    ax = sns.barplot(data=both,
                     x='sample_size_label1',
                     y=col,
                     hue='model_name',
                     alpha=0.8,
                     linewidth=2,
                     palette=palette)  # Use the custom palette
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("% Annotated samples")
    ax.set_ylim([0, 1])
    plt.savefig(os.path.join(dir_fig, f'{col}_figures.png'), bbox_inches='tight')

for col in columns_to_keep[2:]:
    plot_metrics(col)

#plt.show()