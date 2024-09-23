# #import both metrics (smallcnn and gan) for ploting
# l = list()
# #path = 'D:\\OneDrive - University of Illinois - Urbana\\TF\\Data1\\GANS_FL\\output\\head50\\'
# path = 'D:\\OneDrive - University of Illinois - Urbana\\TF\\Data\\GANS_FL\\output\\head50\\'

# for file in [f for f in os.listdir(path) if '_METRICS.csv' in f]:
# #for file in [f for f in os.listdir(path) if '2019class11111.csv' in f]:
#     print(file)
#     #fname = file.split('_')[0]
#     l.append(pd.read_csv(path + file, sep='\t',header=0,index_col=0))

# df = pd.concat(l,ignore_index=True)
# df.tail()
# pd.set_option('display.max_rows', 100)

# both=df
# both

# ##N SAMPLE TO % SAMPLE
# conditions = [
#     (both['label'] == 30),
#     (both['label'] == 32),
#     (both['label'] == 60),
#     (both['label'] == 100),
#     (both['label'] == 300),
#     (both['label'] == 900),
#     (both['label'] == 1800),
#     (both['label'] == 2400),
#     (both['label'] == 2800),
#     (both['label'] == 3000),
# ]
         
# conditions1 = [
#     (both['model'] == 'KNN'),
#     (both['model'] == 'RF'),
#     (both['model'] == 'SMALL_CNN'),
#     (both['model'] == 'TRANSF_CNN1'),
#     (both['model'] == 'GAN'),


# ]
       
# # create a list of the values we want to assign for each condition
# values = [1,1,2,3,10,30,60,80,100,100]
# values1 = ['KNN','RF','CNN','ResNet-50','ESGAN']

# both['label1'] = np.select(conditions, values)
# both['Models'] = np.select(conditions1, values1)





# cats =['KNN','RF','CNN','ResNet-50','ESGAN']
# both['Models'] = pd.CategoricalIndex(both['Models'], ordered=True,categories=cats)
# both = both.sort_values(['label1'],ascending=[True])
# both
# both = both.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
# both


# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# plt.figure(figsize=(14, 10))
# sns.set(font_scale=1.9)
# sns.set_style("ticks")

# def plot_metrics(col):
#     plt.figure()
#     ax = sns.barplot(data=both.iloc[:, 1:6],
#                     x=both.iloc[:, 8],
#                     y=col,
#                     hue=both['Models'],
#                     alpha=0.8,
#                     linewidth=2,
#                     palette=palette)  # Use the custom palette
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#     ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     ax.set_xlabel("% Annotated samples")
#     ax.set_ylim([0, 1])
#     plt.savefig(f'{col}_COLORPALETTE1111.png',bbox_inches='tight')

# for col in both.iloc[:, 1:6].columns:
#     plot_metrics(col)

# plt.show()


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import glob
# import os

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# # Set output directory
# output_dir = '../output'

# def plot_metrics(col, data):
#     plt.figure()
#     ax = sns.barplot(data=data,
#                      x=data['label1'],
#                      y=col,
#                      hue=data['Models'],
#                      alpha=0.8,
#                      linewidth=2,
#                      palette=palette)  # Use the custom palette
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#     ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     ax.set_xlabel("% Annotated samples")
#     ax.set_ylim([0, 1])
#     plt.savefig(os.path.join(output_dir, f'{col}_bars.png'), bbox_inches='tight')

# def main():
#     path = '../output'
    
#     # Load each CSV file and concatenate them into one DataFrame
#     l = []
#     for file in [f for f in os.listdir(path) if '_METRICS.csv' in f]:
#         print(file)
#         l.append(pd.read_csv(os.path.join(path, file), sep='\t', header=0, index_col=0))
    
#     both = pd.concat(l, ignore_index=True)
    
#     print(both.head())
    
#     ##N SAMPLE TO % SAMPLE
#     conditions = [
#         (both['label'] == 30),
#         (both['label'] == 32),
#         (both['label'] == 60),
#         (both['label'] == 100),
#         (both['label'] == 300),
#         (both['label'] == 900),
#         (both['label'] == 1800),
#         (both['label'] == 2400),
#         (both['label'] == 2800),
#         (both['label'] == 3000),
#     ]
            
#     conditions1 = [
#         (both['model'] == 'KNN'),
#         (both['model'] == 'RF'),
#         (both['model'] == 'SMALL_CNN'),
#         (both['model'] == 'TRANSF_CNN1'),
#         (both['model'] == 'GAN'),


#     ]
        
#     # create a list of the values we want to assign for each condition
#     values = [1,1,2,3,10,30,60,80,100,100]
#     values1 = ['KNN','RF','CNN','ResNet-50','ESGAN']

#     both['label1'] = np.select(conditions, values)
#     both['Models'] = np.select(conditions1, values1)
    
#     # Define categories and sort the DataFrame
#     cats = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
#     both['Models'] = pd.CategoricalIndex(both['Models'], ordered=True, categories=cats)
#     both = both.sort_values(['label1'], ascending=[True])
    
#     # Rename columns
#     both = both.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
    
#     plt.figure(figsize=(14, 10))
#     sns.set(font_scale=1.9)
#     sns.set_style("ticks")
    
#     # Iterate over the columns and call plot_metrics
#     for col in both[['F1 score', 'Accuracy', 'precision', 'recall']].columns:
#         plot_metrics(col, both)
    
#     plt.show()

# if __name__ == "__main__":
#     main()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import glob
# import os

# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# # Set output directory
# output_dir = '../output'

# def plot_metrics(col, data, errors):
#     plt.figure()
#     ax = sns.barplot(data=data,
#                      x=data['label1'],
#                      y=col,
#                      hue=data['Models'],
#                      alpha=0.8,
#                      linewidth=2,
#                      palette=palette,
#                      errorbar=None)  # Disable built-in confidence intervals
#     # Add error bars manually
#     for i, bar in enumerate(ax.patches):
#         model = data['Models'].iloc[i // len(data['label1'].unique())]
#         label = data['label1'].iloc[i % len(data['label1'].unique())]
#         error = errors[(errors['Models'] == model) & (errors['label1'] == label)][col].values[0]
#         #ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', c='black', capsize=5)
#         ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', c='black', capsize=bar.get_width() * 5)
    
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#     ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     ax.set_xlabel("% Annotated samples")
#     ax.set_ylim([0, 1])
#     plt.savefig(os.path.join(output_dir, f'{col}_bars.png'), bbox_inches='tight')

# def main():
#     path = '../output'
    
#     # Load each CSV file and concatenate them into one DataFrame
#     l = []
#     for file in [f for f in os.listdir(path) if '_METRICS.csv' in f]:
#         print(file)
#         l.append(pd.read_csv(os.path.join(path, file), sep='\t', header=0, index_col=0))
    
#     both = pd.concat(l, ignore_index=True)
    
#     print(both.head())
    
#     # Load error values from _ERROR.csv files
#     ll = []
#     for file in [f for f in os.listdir(path) if '_ERROR.csv' in f]:
#         print(file)
#         ll.append(pd.read_csv(os.path.join(path, file), sep=',', header=0, index_col=0))
    
#     errors = pd.concat(ll, ignore_index=True)
#     print(errors.head())
    
#     # Rename columns in both and errors DataFrames
#     both = both.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
#     errors = errors.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
    
#     # Define conditions for labels and models
#     conditions = [
#         (both['label'] == 30),
#         (both['label'] == 32),
#         (both['label'] == 60),
#         (both['label'] == 100),
#         (both['label'] == 300),
#         (both['label'] == 900),
#         (both['label'] == 1800),
#         (both['label'] == 2400),
#         (both['label'] == 2800),
#         (both['label'] == 3000),
#     ]
    
#     conditions1 = [
#         (both['model'] == 'KNN'),
#         (both['model'] == 'RF'),
#         (both['model'] == 'SMALL_CNN'),
#         (both['model'] == 'TRANSF_CNN1'),
#         (both['model'] == 'GAN'),
#     ]
    
#     # Create a list of the values we want to assign for each condition
#     values = [1, 1, 2, 3, 10, 30, 60, 80, 100, 100]
#     values1 = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
    
#     both['label1'] = np.select(conditions, values)
#     both['Models'] = np.select(conditions1, values1)
    
#     # Apply the same conditions to the errors DataFrame
#     errors['label1'] = np.select(conditions, values)
#     errors['Models'] = np.select(conditions1, values1)
    
#     # Define categories and sort the DataFrame
#     cats = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
#     both['Models'] = pd.CategoricalIndex(both['Models'], ordered=True, categories=cats)
#     both = both.sort_values(['label1'], ascending=[True])
    
#     plt.figure(figsize=(14, 10))
#     sns.set(font_scale=1.9)
#     sns.set_style("ticks")
    
#     # Iterate over the columns and call plot_metrics
#     for col in both[['F1 score', 'Accuracy', 'precision', 'recall']].columns:
#         plot_metrics(col, both, errors)
    
#     #plt.show()

# if __name__ == "__main__":
#     main()
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import os
# from tabulate import tabulate  # Import tabulate


# # Define a custom palette with hexadecimal color codes
# palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# # Set output directory
# output_dir = '../output'

# def plot_metrics(col, error_col, data):
#     plt.figure()
#     ax = sns.barplot(data=data,
#                      x='label1',
#                      y=col,
#                      hue='Models',
#                      alpha=0.8,
#                      linewidth=2,
#                      palette=palette,
#                      errorbar=None)  # Disable built-in confidence intervals
#     # Add error bars manually
#     for bar, hue in zip(ax.patches, data['Models']):
#         # Get the corresponding row in the DataFrame
#         label1 = bar.get_x() + bar.get_width() / 2
#         height = bar.get_height()
#         model = hue
#         row = data[(data['label1'] == label1) & (data['Models'] == model) & (data[col] == height)]
#         print(f"Bar: {bar}, Label1: {label1}, Model: {model}, Row: {row}")
#         if not row.empty:
#             error = row.iloc[0][error_col]
#             print(f"Error for {col} at Label1: {label1}, Model: {model} is {error}")
#             ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', c='black', capsize=bar.get_width() * 5)
#         else:
#             print('empty rows')
    
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#     ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
#     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#     ax.set_xlabel("% Annotated samples")
#     ax.set_ylim([0, 1])
#     plt.savefig(os.path.join(output_dir, f'{col}_bars.png'), bbox_inches='tight')

# def main():
#     path = '../output'
    
#     # Load each CSV file and concatenate them into one DataFrame
#     l = []
#     for file in [f for f in os.listdir(path) if '_METRICS.csv' in f]:
#         print(file)
#         l.append(pd.read_csv(os.path.join(path, file), sep='\t', header=0, index_col=0))
    
#     both = pd.concat(l, ignore_index=True)
    
#     print("Both DataFrame:")
#     print(both.head())
    
#     # Load error values from _ERROR.csv files
#     ll = []
#     for file in [f for f in os.listdir(path) if '_ERROR.csv' in f]:
#         print(file)
#         ll.append(pd.read_csv(os.path.join(path, file), sep=',', header=0, index_col=0))
    
#     errors = pd.concat(ll, ignore_index=True)
#     print("Errors DataFrame:")
#     print(errors.head())
    
#     # Rename columns in both and errors DataFrames
#     both = both.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
#     errors = errors.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
    
#     # Define conditions for labels and models
#     conditions = [
#         (both['label'] == 30),
#         (both['label'] == 32),
#         (both['label'] == 60),
#         (both['label'] == 100),
#         (both['label'] == 300),
#         (both['label'] == 900),
#         (both['label'] == 1800),
#         (both['label'] == 2400),
#         (both['label'] == 2800),
#         (both['label'] == 3000),
#     ]
    
#     conditions1 = [
#         (both['model'] == 'KNN'),
#         (both['model'] == 'RF'),
#         (both['model'] == 'SMALL_CNN'),
#         (both['model'] == 'TRANSF_CNN1'),
#         (both['model'] == 'GAN'),
#     ]
    
#     # Create a list of the values we want to assign for each condition
#     values = [1, 1, 2, 3, 10, 30, 60, 80, 100, 100]
#     values1 = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
    
#     both['label1'] = np.select(conditions, values)
#     both['Models'] = np.select(conditions1, values1)
    
#     # Apply the same conditions to the errors DataFrame
#     errors['label1'] = np.select(conditions, values)
#     errors['Models'] = np.select(conditions1, values1)
    
#     # Merge both and errors DataFrames on 'label1', 'Models', 'label', and 'model'
#     merged = pd.merge(both, errors, on=['label1', 'Models', 'label', 'model'], suffixes=('', '_error'))
    
#     # Define categories and sort the DataFrame
#     cats = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
#     merged['Models'] = pd.Categorical(merged['Models'], ordered=True, categories=cats)
#     merged = merged.sort_values(['label1'], ascending=[True])
    
#     print("Merged DataFrame after adding 'label1' and 'Models':")
#     print(tabulate(merged, headers='keys', tablefmt='psql')) 
    
#     plt.figure(figsize=(14, 10))
#     sns.set(font_scale=1.9)
#     sns.set_style("ticks")
    
#     # Iterate over the columns and call plot_metrics
#     for col in ['F1 score', 'Accuracy', 'precision', 'recall']:
#         plot_metrics(col, f'{col}_error', merged)
    
#     #plt.show()

# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from tabulate import tabulate  # Import tabulate

# Define a custom palette with hexadecimal color codes
palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# Set output directory
output_dir = '../output'

def plot_metrics(col, data):
    plt.figure()

    # Add caps to error bars using the 'capsize' argument
    ax = sns.barplot(data=data,
                     x='label1',
                     y=col,
                     hue='Models',
                     alpha=0.8,
                     linewidth=2,
                     palette=palette,
                     errorbar='sd',
                     capsize=0.1,errwidth=1)  # Add caps to error bars

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("% Annotated samples")
    ax.set_ylim([0, 1])

    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{col}_bars.png'), bbox_inches='tight')

def main():
    path = '../output'
    
    # Load each CSV file and concatenate them into one DataFrame
    l = []
    for file in [f for f in os.listdir(path) if 'METRICS' in f]:
        print(file)
        df = pd.read_csv(os.path.join(path, file), sep='\t', header=0, index_col=0)
        df['source'] = file.split('_')[0]  # Add a column to identify the source file
        l.append(df)
    
    both = pd.concat(l, ignore_index=True)
    
    print("Both DataFrame:")
    print(both.head())
    
    # Rename columns in both DataFrame
    both = both.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
    
    # Define conditions for labels and models
    conditions = [
        (both['label'] == 30),
        (both['label'] == 32),
        (both['label'] == 60),
        (both['label'] == 100),
        (both['label'] == 300),
        (both['label'] == 900),
        (both['label'] == 1800),
        (both['label'] == 2400),
        (both['label'] == 2800),
        (both['label'] == 3000),
    ]
    
    conditions1 = [
        (both['model'] == 'KNN'),
        (both['model'] == 'RF'),
        (both['model'] == 'SMALL_CNN'),
        (both['model'] == 'TRANSF_CNN1'),
        (both['model'] == 'GAN'),
    ]
    
    # Create a list of the values we want to assign for each condition
    values = [1, 1, 2, 3, 10, 30, 60, 80, 100, 100]
    values1 = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
    
    both['label1'] = np.select(conditions, values)
    both['Models'] = np.select(conditions1, values1)
    
    # Define categories and sort the DataFrame
    cats = ['KNN', 'RF', 'CNN', 'ResNet-50', 'ESGAN']
    both['Models'] = pd.Categorical(both['Models'], ordered=True, categories=cats)
    both = both.sort_values(['label1', 'Models'], ascending=[True, True])
    
    print("Both DataFrame after adding 'label1' and 'Models':")
    print(tabulate(both, headers='keys', tablefmt='psql'))  # Print both DataFrame as table
    
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.9)
    sns.set_style("ticks")
    
    # Iterate over the columns and call plot_metrics
    for col in ['F1 score', 'Accuracy', 'precision', 'recall']:
        plot_metrics(col, both)
    
    #plt.show()

if __name__ == "__main__":
    main()
    
    
    
