import pandas as pd
import sys

sys.path.append('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/src')
from feature_extractor import calculate_features

# Load the dataset
data = pd.read_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/trajectories_per_day.csv')

# Convert the Date column to datetime format and create a YearMonth column
data['Date'] = pd.to_datetime(data['Date'])
data['YearMonth'] = data['Date'].dt.to_period('M')

# Perform monthly aggregation
monthly_trajectories = data.groupby(['YearMonth', 'ORIGIN', 'DESTINATION'], as_index=False).agg({
    'Rides planned': 'sum',
    'Final arrival delay': 'sum',
    'Final arrival cancelled': 'sum',
    'Completely cancelled': 'sum',
    'Intermediate arrival delays': 'sum'
})

# Filter out trajectories with fewer than 4 rides planned per month
min_rides_threshold = 4
monthly_trajectories = monthly_trajectories[monthly_trajectories['Rides planned'] >= min_rides_threshold]

# Recalculate the proportion delayed after monthly aggregation
monthly_trajectories['Proportion delayed'] = monthly_trajectories.apply(
    lambda row: 0 if row['Final arrival delay'] == 0 else 
    row['Final arrival delay'] / (row['Rides planned'] - row['Completely cancelled']) 
    if (row['Rides planned'] - row['Completely cancelled']) > 0 else 0, axis=1)


# Rename columns for NetworkX compatibility
monthly_trajectories = monthly_trajectories.rename(columns={'ORIGIN': 'source', 'DESTINATION': 'target'})

# Calculate the 90th percentile threshold
percentile_90 = monthly_trajectories['Proportion delayed'].quantile(0.90)
print(f'90th Percentile Threshold for Significant Delay: {percentile_90}')

# Add a column to indicate if the delay is significant based on the threshold
monthly_trajectories['Significant Delay'] = monthly_trajectories['Proportion delayed'] > percentile_90

monthly_trajectories.to_csv('test.csv')
# Filter the months you are interested in
# filtered_months = ['2019-01', '2019-02', '2019-03']
# filtered_trajectories = monthly_trajectories[monthly_trajectories['YearMonth'].astype(str).isin(filtered_months)]

# Calculate topological features
monthly_features = calculate_features(monthly_trajectories)

# Include the 'Proportion delayed', 'Significant Delay', and other required columns in the features dataset
monthly_features = pd.merge(
    monthly_features, 
    monthly_trajectories[['YearMonth', 'source', 'target', 'Proportion delayed', 'Significant Delay', 'Rides planned', 'Final arrival delay', 'Final arrival cancelled', 'Completely cancelled', 'Intermediate arrival delays']], 
    on=['YearMonth', 'source', 'target'], 
    how='left'
)

# Save the resulting features to a CSV file
monthly_features.to_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/features/monthly_features_per_trajectory.csv', index=False)