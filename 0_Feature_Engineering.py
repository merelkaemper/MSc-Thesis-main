import pandas as pd
import sys

sys.path.append('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/src')
from feature_extractor import calculate_features

# Load the dataset
data_path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/data/monthly_trajectories.csv'
data = pd.read_csv(data_path)

# Convert YearMonth column to datetime format
data['YearMonth'] = pd.to_datetime(data['YearMonth'], format='%Y-%m').dt.to_period('M')

# Calculate features
features_df = calculate_features(data)

# Merge with the original data to include labels and additional attributes
features_df = pd.merge(
    features_df, 
    data[['YearMonth', 'source', 'target', 'Proportion delayed', 'Significant Delay', 'Rides planned', 'Final arrival delay', 'Final arrival cancelled', 'Completely cancelled', 'Intermediate arrival delays']], 
    on=['YearMonth', 'source', 'target'], 
    how='right'  
)

# Save the resulting features to a CSV file
output_path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/features/monthly_features_per_trajectory.csv'
features_df.to_csv(output_path, index=False)
#features_df.isna().sum()

print("Feature engineering completed and saved to:", output_path)