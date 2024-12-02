
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from tabulate import tabulate  # Import tabulate

# Define a custom palette with hexadecimal color codes
palette = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#00008B']

# Set output directory
output_dir = '/output'

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
    plt.savefig(os.path.join(output_dir, f'{col}_bars11.png'), bbox_inches='tight')

def main():
    path = '../output/output'
    
    # Load each CSV file and concatenate them into one DataFrame
    l = []
    for file in [f for f in os.listdir(path) if 'metrics' in f]:
        print(file)
        #df = pd.read_csv(os.path.join(path, file), sep=',', header=0, index_col=0)
        df = pd.read_csv(os.path.join(path, file), sep=',', header=0)
        print(df.head())
        df['source'] = file.split('_')[0]  # Add a column to identify the source file
        l.append(df)
    
    both = pd.concat(l, ignore_index=True)
    both = pd.concat(l)
    
    print("Both DataFrame:")
    print(both.head(200))
    
    # Rename columns in both DataFrame
    both = both.rename(columns={'f1': 'F1 score', 'accuracy': 'Accuracy'})
    
    # Define conditions for labels and models
    conditions = [
        (both['label'] == 30),
        (both['label'] == 60),
        (both['label'] == 100),
        (both['label'] == 300),
        (both['label'] == 900),
        (both['label'] == 1800),
        (both['label'] == 2400),
        (both['label'] == 3000),
    ]
    
    conditions1 = [
        (both['model'] == 'KNN'),
        (both['model'] == 'RF'),
        (both['model'] == 'SMALLCNN'),
        (both['model'] == 'TRANSFCNN'),
        (both['model'] == 'ESGAN'),
    ]
    
    # Create a list of the values we want to assign for each condition
    values = [1, 2, 3, 10, 30, 60, 80, 100]
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
    for col in ['F1 score', 'Accuracy']:
        plot_metrics(col, both)
    
    #plt.show()

if __name__ == "__main__":
    main()