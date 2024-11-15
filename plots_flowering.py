# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Ensure the output directory exists
# dir_fig = '../output/predictions_09222024'
# os.makedirs(dir_fig, exist_ok=True)

# # Load the CSV files into DataFrames
# benchmark_df = pd.read_csv(os.path.join(dir_fig, 'benchmarks_models_predictions_and_y_test.csv'))
# best_gan_df = pd.read_csv(os.path.join(dir_fig, 'gan_models_predictions_and_y_test.csv'))
# benchmark_df = benchmark_df[benchmark_df['iteration_n'] == 0]

# # Define the columns to keep
# columns_to_keep = ['model_name', 'sample_size_label1', 'pred_test', 'y_test', 'grouping']

# # Subset the columns needed from each DataFrame
# benchmark_df = benchmark_df[columns_to_keep]
# best_gan_df = best_gan_df[columns_to_keep]

# both = pd.concat([benchmark_df, best_gan_df], ignore_index=True)

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# def calculate_metrics(group):
#     accuracy = accuracy_score(group['y_test'], group['pred_test'])
#     precision = precision_score(group['y_test'], group['pred_test'], average='weighted')
#     recall = recall_score(group['y_test'], group['pred_test'], average='weighted')
#     f1 = f1_score(group['y_test'], group['pred_test'], average='weighted')
#     return pd.Series({
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     })

# # Group by model_name, sample_size_label1, and grouping and calculate metrics
# grouped_metrics = both.groupby(['model_name', 'sample_size_label1', 'grouping']).apply(calculate_metrics).reset_index()

# # Define the order of models
# model_order = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']

# # Define y-axis labels for each metric
# y_axis_labels = {
#     'accuracy': 'Accuracy',
#     'precision': 'Precision',
#     'recall': 'Recall',
#     'f1': 'F1 Score'
# }

# def plot_metrics(col, flowering_date):
#     data = grouped_metrics[grouped_metrics['grouping'] == flowering_date]
#     # Set plot settings
#     plt.figure(figsize=(16, 10))
#     sns.set_theme(font_scale=2.9)
#     sns.set_style("ticks")
    
#     ax = sns.barplot(data=data,
#                      x='sample_size_label1',
#                      y=col,
#                      hue='model_name',
#                      hue_order=model_order,  # Set the order of models
#                      alpha=0.8,
#                      linewidth=2,
#                      palette=palette)  # Use the custom palette
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#     ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    
#     # Move the legend and set its title
#     legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Models')
    
#     ax.set_xlabel("% Annotated samples")
#     ax.set_ylabel(y_axis_labels[col])  # Set the y-axis label based on the metric
#     ax.set_ylim([0, 1])
    
#     # Add text annotations
#     sample_size_label = data['sample_size_label1'].unique()[0]  # Get the sample size label
#     #plt.text(0.5, -0.1, f'Flowering time = {flowering_date}', fontsize=20, transform=ax.transAxes, color='black')
#     plt.title(f'Flowering time = {flowering_date}', fontsize=20, color='black')

#     plt.tight_layout()  # Adjust the layout to fit elements properly
#     plt.savefig(os.path.join(dir_fig, f'{col}_flowering_date_{flowering_date}_figures.png'), bbox_inches='tight')
#     plt.close()

# flowering_dates = grouped_metrics['grouping'].unique()
# for col in ['accuracy', 'precision', 'recall', 'f1']:
#     for flowering_date in flowering_dates:
#         plot_metrics(col, flowering_date)
        
        
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# # Ensure the output directory exists
# dir_fig = '../output/predictions_09222024'
# os.makedirs(dir_fig, exist_ok=True)

# # Load the CSV files into DataFrames
# benchmark_df = pd.read_csv(os.path.join(dir_fig, 'benchmarks_models_predictions_and_y_test.csv'))
# best_gan_df = pd.read_csv(os.path.join(dir_fig, 'gan_models_predictions_and_y_test.csv'))
# benchmark_df = benchmark_df[benchmark_df['iteration_n'] == 0]

# # Define the columns to keep
# columns_to_keep = ['model_name', 'sample_size_label1', 'pred_test', 'y_test', 'grouping']

# # Subset the columns needed from each DataFrame
# benchmark_df = benchmark_df[columns_to_keep]
# best_gan_df = best_gan_df[columns_to_keep]

# # Concatenate the DataFrames
# both = pd.concat([benchmark_df, best_gan_df], ignore_index=True)

# # Ensure 'grouping' is treated as numerical values
# both['grouping'] = pd.to_numeric(both['grouping'], errors='coerce')

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B', '#000000']

# # Define the order of models
# model_order = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']

# # Function to plot bar plots for pred_test
# def plot_pred_test(sample_size_label):
#     data = both[both['sample_size_label1'] == sample_size_label]
    
#     # Count the occurrences of the event 1 in pred_test
#     event_counts = data[data['pred_test'] == 1].groupby(['grouping', 'model_name']).size().reset_index(name='event_count')
    
#     # Count the occurrences of the event 1 in y_test for a reference model (e.g., 'KNN')
#     reference_model = 'KNN'
#     ground_truth_counts = data[(data['model_name'] == reference_model) & (data['y_test'] == 1)].groupby('grouping').size().reset_index(name='event_count')
#     ground_truth_counts['model_name'] = 'Ground-Truth'
    
#     # Append ground-truth counts to event_counts
#     event_counts = pd.concat([event_counts, ground_truth_counts], ignore_index=True)
    
#     # Debugging: Print the shape and content of event_counts
#     print(f"Event counts for sample_size_label={sample_size_label}:")
#     print(event_counts.head())
#     print(f"Shape of event_counts: {event_counts.shape}")
    
#     # Skip plotting if event_counts is empty
#     if event_counts.empty:
#         print(f"No data to plot for sample_size_label={sample_size_label}. Skipping.")
#         return
    
#     # Set plot settings
#     plt.figure(figsize=(14, 10))
#     sns.set_theme(font_scale=2.9)
#     sns.set_style("ticks")
    
#     ax = sns.barplot(data=event_counts,
#                      x='grouping',
#                      y='event_count',
#                      hue='model_name',
#                      hue_order=model_order + ['Ground-Truth'],  # Include Ground-Truth in the order
#                      alpha=0.8,
#                      linewidth=2,
#                      palette=palette + ['#FF0000'])  # Add a color for Ground-Truth
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    
#     # Move the legend and set its title
#     legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Models')
    
#     ax.set_xlabel("Flowering time (Julian Dates)")
#     ax.set_ylabel("N flowering events detected")  # Set the y-axis label
    
#     # Add polynomial trend line for ESGAN
#     esgan_data = event_counts[event_counts['model_name'] == 'Ground-Truth']
#     if not esgan_data.empty:
#         # Get the x-axis positions of the groupings
#         x_positions = esgan_data['grouping'].values
#         y_values = esgan_data['event_count'].values
        
#         # Debugging: Print x_positions and y_values
#         print(f"x_positions for Ground-truth: {x_positions}")
#         print(f"y_values for Ground-truth: {y_values}")
        
#         if len(x_positions) >= 2:
#             # Fit a polynomial of degree 2 (quadratic trend line)
#             poly_coeffs = np.polyfit(x_positions, y_values, 2)
#             poly_fit = np.polyval(poly_coeffs, x_positions)
            
#             # Debugging: Print poly_fit values
#             print(f"poly_fit values for Ground-truth: {poly_fit}")
            
#             # Get the actual x-tick positions used by the bar plot
#             xticks = ax.get_xticks()
#             xticklabels = [int(label.get_text()) for label in ax.get_xticklabels()]
#             print(f"xticks: {xticks}")
#             print(f"xticklabels: {xticklabels}")
            
#             # Map the x_positions to the actual x-tick positions
#             x_positions_mapped = [xticks[xticklabels.index(x)] for x in x_positions]
#             print(f"x_positions_mapped for Ground-truth: {x_positions_mapped}")
            
#             ax.plot(x_positions_mapped, poly_fit, label='Ground-truth Trend', linestyle='--', color='black')
    
#     plt.tight_layout()  # Adjust the layout to fit elements properly
#     plt.savefig(os.path.join(dir_fig, f'pred_test_count_{sample_size_label}.png'), bbox_inches='tight')
#     plt.close()

# # Generate plots for each sample size label
# sample_size_labels = both['sample_size_label1'].unique()
# for sample_size_label in sample_size_labels:
#     plot_pred_test(sample_size_label)

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# # Ensure the output directory exists
# dir_fig = '../output/predictions_09222024'
# os.makedirs(dir_fig, exist_ok=True)

# # Load the CSV files into DataFrames
# benchmark_df = pd.read_csv(os.path.join(dir_fig, 'benchmarks_models_predictions_and_y_test.csv'))
# best_gan_df = pd.read_csv(os.path.join(dir_fig, 'gan_models_predictions_and_y_test.csv'))
# benchmark_df = benchmark_df[benchmark_df['iteration_n'] == 0]

# # Define the columns to keep
# columns_to_keep = ['model_name', 'sample_size_label1', 'pred_test', 'y_test', 'grouping']

# # Subset the columns needed from each DataFrame
# benchmark_df = benchmark_df[columns_to_keep]
# best_gan_df = best_gan_df[columns_to_keep]

# # Concatenate the DataFrames
# both = pd.concat([benchmark_df, best_gan_df], ignore_index=True)

# # Ensure 'grouping' is treated as numerical values
# both['grouping'] = pd.to_numeric(both['grouping'], errors='coerce')

# print(both)

# # Exclude rows where 'grouping' is 296
# both = both[both['grouping'] != 296]

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B', '#000000']

# # Define the order of models
# model_order = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']

# # Define dash patterns for all models
# dashes = {
#     'KNN': '',
#     'RF': '',
#     'CNN': '',
#     'ResNet-50': '',
#     'ESGAN': '',
#     'Ground-Truth': (5, 2)  # Set Ground-Truth line to be dashed
# }

# # List of sample sizes
# l = [32, 64, 94, 314, 941, 1884, 2509, 3137]

# # Function to plot line plots for pred_test
# def plot_pred_test(sample_size_label, n_value):
#     data = both[both['sample_size_label1'] == sample_size_label]
    
#     # Count the occurrences of the event 1 in pred_test
#     event_counts = data[data['pred_test'] == 1].groupby(['grouping', 'model_name']).size().reset_index(name='event_count')
    
#     # Count the occurrences of the event 1 in y_test for a reference model (e.g., 'KNN')
#     reference_model = 'KNN'
#     ground_truth_counts = data[(data['model_name'] == reference_model) & (data['y_test'] == 1)].groupby('grouping').size().reset_index(name='event_count')
#     ground_truth_counts['model_name'] = 'Ground-Truth'
    
#     # Append ground-truth counts to event_counts
#     event_counts = pd.concat([event_counts, ground_truth_counts], ignore_index=True)
    
#     # Debugging: Print the shape and content of event_counts
#     print(f"Event counts for sample_size_label={sample_size_label}:")
#     print(event_counts.head())
#     print(f"Shape of event_counts: {event_counts.shape}")
    
#     # Skip plotting if event_counts is empty
#     if event_counts.empty:
#         print(f"No data to plot for sample_size_label={sample_size_label}. Skipping.")
#         return
    
#     # Set plot settings
#     plt.figure(figsize=(16, 10))
#     sns.set_theme(font_scale=3.2)
#     sns.set_style("ticks")
    
#     ax = sns.lineplot(data=event_counts,
#                       x='grouping',
#                       y='event_count',
#                       hue='model_name',
#                       hue_order=model_order + ['Ground-Truth'],  # Include Ground-Truth in the order
#                       palette=palette,  # Use the updated palette
#                       markers=True,  # Add markers to the lines
#                       style='model_name',  # Different styles for different models
#                       dashes=dashes,  # Set dash patterns for all models
#                       linewidth=3.5,  # Increase line thickness
#                       markersize=20)  # Increase marker size
    
#     # Set x-axis ticks from 240 to 300, separated by 5 days
#     x_ticks = np.arange(245, 280, 5)
#     plt.xticks(x_ticks)
    
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    
#     # Move the legend and set its title
#     legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Models')
    
#     ax.set_xlabel("Flowering time (Julian Dates)")
#     ax.set_ylabel("N Flowering events detected")  # Set the y-axis label
    
#     # Add text annotations
#     plt.text(0.8, 0.25, f'{sample_size_label}%', fontsize=40, transform=ax.transAxes)
#     plt.text(0.8, 0.20, f'(n={n_value})', fontsize=30, transform=ax.transAxes)
    
#     plt.tight_layout()  # Adjust the layout to fit elements properly
#     plt.savefig(os.path.join(dir_fig, f'pred_test_count_{sample_size_label}.png'), bbox_inches='tight')
#     plt.close()

# # Generate plots for each sample size label
# sample_size_labels = both['sample_size_label1'].unique()
# for sample_size_label, n_value in zip(sample_size_labels, l):
#     plot_pred_test(sample_size_label, n_value)

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # # Ensure the output directory exists
# dir_fig = '../output/predictions_09222024'

# # # Load the CSV files into DataFrames
# training_times_df = pd.read_csv(os.path.join(dir_fig, 'training_times_benchmarks.csv'))

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

# # Split the Model column into Annotated samples and Model name
# # Split the Model column into Annotated samples and Model name

# print(training_times_df)
# training_times_df['Annotated samples'] = training_times_df['Model'].apply(lambda x: x.split('_')[0])
# training_times_df['Model'] = training_times_df['Model'].apply(lambda x: x.split('_')[1])

# # Update the Model names
# training_times_df['Model'] = training_times_df['Model'].apply(update_model_name)

# # Convert Training Time to minutes, seconds, and milliseconds
# training_times_df['Training Time (min:sec.ms)'] = training_times_df['Training Time'].apply(
#     lambda x: f"{int(x // 60)}:{int(x % 60):02d}.{int((x * 1000) % 1000):03d}"
# )

# print(training_times_df)

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B', '#000000']

# # Define the order of models
# model_order = ['KNN', 'RF', 'CNN', 'ResNet-50']

# # Create the bar plot
# plt.figure(figsize=(14, 10))
# sns.set_theme(style="whitegrid")
# ax = sns.barplot(x='Annotated samples', y='Training Time', hue='Model', data=training_times_df, palette=palette, hue_order=model_order)

# # Add labels and title
# ax.set_xlabel('% Annotated samples')
# ax.set_ylabel('Training time (seconds)')
# ax.set_title('Training Time by Model and Annotated Samples')

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# # # Add text annotations for detailed time format
# # for p in ax.patches:
# #     height = p.get_height()
# #     if not pd.isna(height):
# #         ax.text(
# #             p.get_x() + p.get_width() / 2.,
# #             height,
# #             f'{int(height // 60)}:{int(height % 60):02d}.{int((height * 1000) % 1000):03d}',
# #             ha='center'
# #         )

# # Move the legend and set its title
# # Move the legend and set its title
# legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Models')

# # Adjust layout to avoid UserWarning
# plt.tight_layout(rect=[0, 0, 0.85, 1])
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the output directory exists
dir_fig = '../output/predictions_09222024'
os.makedirs(dir_fig, exist_ok=True)

# Load the CSV files into DataFrames
training_times_benchmarks_df = pd.read_csv(os.path.join(dir_fig, 'training_times_benchmarks.csv'))
training_times_gan_df = pd.read_csv(os.path.join(dir_fig, 'training_times_gan.csv'))

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
    elif 'ESGAN' in model_name:
        return 'ESGAN'
    else:
        return 'Unknown'

# Process the benchmarks DataFrame
training_times_benchmarks_df['Annotated samples'] = training_times_benchmarks_df['Model'].apply(lambda x: x.split('_')[0])
training_times_benchmarks_df['Model'] = training_times_benchmarks_df['Model'].apply(lambda x: x.split('_')[1])
training_times_benchmarks_df['Model'] = training_times_benchmarks_df['Model'].apply(update_model_name)

# Create a mapping dictionary for sample sizes to percentages
sample_size_to_percent = {
    '30': '1',
    '60': '2',
    '100': '3',
    '300': '10',
    '900': '30',
    '1800': '60',
    '2400': '80',
    '3000': '100'
}

# Generate the Percent Annotated samples column for benchmarks DataFrame
training_times_benchmarks_df['Percent Annotated samples'] = training_times_benchmarks_df['Annotated samples'].map(sample_size_to_percent)

# Process the GAN DataFrame
training_times_gan_df['Annotated samples'] = training_times_gan_df['Sample Size'].astype(str)
training_times_gan_df['Model'] = training_times_gan_df['model']

# Generate the Percent Annotated samples column for GAN DataFrame
training_times_gan_df['Percent Annotated samples'] = training_times_gan_df['Annotated samples'].map(sample_size_to_percent)

# Update the Model names in GAN DataFrame
training_times_gan_df['Model'] = training_times_gan_df['Model'].apply(update_model_name)

# Combine both DataFrames
combined_df = pd.concat([training_times_benchmarks_df, training_times_gan_df], ignore_index=True)

# Ensure 'Percent Annotated samples' is treated as a categorical variable with a specific order
percent_annotated_samples_order = ['1', '2', '3', '10', '30', '60', '80', '100']
combined_df['Percent Annotated samples'] = pd.Categorical(combined_df['Percent Annotated samples'], categories=percent_annotated_samples_order, ordered=True)

# Convert Training Time to minutes, seconds, and milliseconds
combined_df['Training Time (min:sec.ms)'] = combined_df['Training Time'].apply(
    lambda x: f"{int(x // 60)}:{int(x % 60):02d}.{int((x * 1000) % 1000):03d}"
)

# Define a custom palette with hexadecimal color codes
palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B', '#000000']

# Define the order of models
model_order = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']

# Set the font size globally
plt.rcParams.update({'font.size': 2.9})

# Create the bar plot
plt.figure(figsize=(16, 10))

sns.set_theme(style="whitegrid", font_scale=2.0)
ax = sns.barplot(x='Percent Annotated samples', y='Training Time', hue='Model', data=combined_df, palette=palette, hue_order=model_order)

# Add labels and title
ax.set_xlabel('% Annotated samples')
ax.set_ylabel('Training time (seconds)')
ax.set_title('Training Time by Model and Annotated Samples')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Move the legend and set its title
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Models')

# Adjust layout to avoid UserWarning
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save the plot to the specified directory
plt.savefig(os.path.join(dir_fig, 'training_time_by_model_and_annotated_samples1.png'), bbox_inches='tight')

# Show the plot
plt.show()