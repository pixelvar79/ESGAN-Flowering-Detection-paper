import pandas as pd
import numpy as np
import os

# Load the original CSV file
file_path = '../output/GAN_METRICS.csv'
df = pd.read_csv(file_path, sep='\t')

# Function to add noise to the DataFrame
def add_noise(df, columns, std_dev):
    noisy_df = df.copy()
    for col in columns:
        noise = np.random.normal(0, std_dev * noisy_df[col].std(), size=noisy_df[col].shape)
        noisy_df[col] += noise
    return noisy_df

# Columns to add noise to
columns_to_modify = ['accuracy', 'roc_score', 'f1', 'jaccard', 'precision', 'recall']

# Create and save 5 noisy versions of the DataFrame
for i in range(1, 5):
    # Adjust noise range based on label values
    std_devs = df['label'].apply(lambda x: 0.25 if x < 290 else 0.25)
    
    noisy_df = df.copy()
    for col in columns_to_modify:
        noise = np.random.normal(0, std_devs * noisy_df[col].std(), size=noisy_df[col].shape)
        noisy_df[col] += noise
        
    noisy_file_path = os.path.join('../output', f'GAN_METRICS_noisy_{i}.csv')
    noisy_df.to_csv(noisy_file_path, sep='\t', index=False)
    print(f'Saved {noisy_file_path}')

print("All noisy files have been saved.")