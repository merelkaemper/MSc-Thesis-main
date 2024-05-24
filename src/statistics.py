import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/trajectories_per_day.csv')

# Convert the Date column to datetime format
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

monthly_trajectories['Proportion delayed'] = monthly_trajectories.apply(
    lambda row: 0 if row['Final arrival delay'] == 0 else 
    row['Final arrival delay'] / (row['Rides planned'] - row['Completely cancelled']) 
    if (row['Rides planned'] - row['Completely cancelled']) > 0 else 0, axis=1
)

# Calculate basic statistics
mean_proportion = monthly_trajectories['Proportion delayed'].mean()
median_proportion = monthly_trajectories['Proportion delayed'].median()
std_proportion = monthly_trajectories['Proportion delayed'].std()

print(f'Mean Proportion Delayed: {mean_proportion}')
print(f'Median Proportion Delayed: {median_proportion}')
print(f'Standard Deviation of Proportion Delayed: {std_proportion}')

# Calculate the 90th percentile threshold
percentile_90 = monthly_trajectories['Proportion delayed'].quantile(0.90)
print(f'90th Percentile Threshold for Significant Delay: {percentile_90}')



