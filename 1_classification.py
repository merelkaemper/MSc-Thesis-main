import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.stats as ss
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import pickle

sys.path.append('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/src')
from plot_style import *

# Define the best hyperparameters for XGBoost
best_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42
}

# Split the dataset into training and testing set based on edges 
def get_edge_slice(data, f_train_e=0.7, seed=30):
    edges = list(zip(data['source'], data['target']))  # Get unique edges as tuples of (source, target)
    random.seed(seed)  # Set the random seed for reproducibility
    edge_train = set(random.sample(edges, int(f_train_e * len(edges))))  # Randomly select training edges
    edge_test = set(edges) - edge_train  # The rest are testing edges
    df_train = data[data.apply(lambda row: (row['source'], row['target']) in edge_train, axis=1)].drop(columns=['source', 'target'])  # Training set
    df_test = data[data.apply(lambda row: (row['source'], row['target']) in edge_test, axis=1)].drop(columns=['source', 'target'])  # Testing set
    return df_train, df_test

# Convert DataFrame to feature matrix X and target vector y
def df_to_XY(df, features, target='Significant Delay'):
    X = df[features].to_numpy()
    y = df[target].astype(int).to_numpy()  # Ensure the target is binary
    return X, y

# Train the XGBoost model with undersampling to handle class imbalance
def train_model(X_train, y_train):
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)  # Apply undersampling to the training data
    model = XGBClassifier(**best_params)  # Initialize the XGBoost classifier with the best parameters
    model.fit(X_train, y_train)  # Train the model on the undersampled data
    return model

# Perform simultaneous testing by training and testing on the same year/month
def simultaneous_test(df_train, df_test, features, save=True, name=None):
    if name is None:
        name = 'simultaneous_test'

    year_month_list = df_train['YearMonth'].unique()  # Get unique year-month periods
    res_df_test = df_test.copy()
    res_df_test['simultaneous_pred'] = np.nan
    res_df_test['simultaneous_null'] = np.nan

    for year_month in tqdm(year_month_list):
        train_subset = df_train[df_train['YearMonth'] == year_month]  # Training subset for the current period
        test_subset = df_test[df_test['YearMonth'] == year_month]  # Testing subset for the same period

        X_train, y_train = df_to_XY(train_subset, features)  # Prepare training data
        X_test, y_test = df_to_XY(test_subset, features)  # Prepare testing data

        if len(y_train) == 0 or len(y_test) == 0:
            continue  # Skip if either the training or testing data is empty for the period

        model = train_model(X_train, y_train)  # Train the model
        model_null = train_model(X_train, np.random.permutation(y_train))  # Train a null model

        y_pred = model.predict(X_test)
        y_pred_null = model_null.predict(X_test)

        res_df_test.loc[res_df_test['YearMonth'] == year_month, 'simultaneous_pred'] = y_pred
        res_df_test.loc[res_df_test['YearMonth'] == year_month, 'simultaneous_null'] = y_pred_null

    if save:
        res_df_test.to_csv(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/results/{name}.csv', index=False)  # Save results
    return res_df_test

# Perform non-simultaneous testing by training on one period and testing on subsequent periods
def nonsimultaneous_test(df_train, df_test, features, save=True, name=None):
    if name is None:
        name = 'nonsimultaneous_test'

    year_month_list = df_train['YearMonth'].unique()  # Get unique year-month periods
    results = []

    for i, train_period in enumerate(tqdm(year_month_list)):
        train_subset = df_train[df_train['YearMonth'] == train_period]  # Training subset for the current period

        X_train, y_train = df_to_XY(train_subset, features)  # Prepare training data

        if len(y_train) == 0:
            continue  # Skip if the training data is empty for the period

        model = train_model(X_train, y_train)  # Train the model

        for test_period in year_month_list[i:]:
            test_subset = df_test[df_test['YearMonth'] == test_period]  # Testing subset for the subsequent period
            X_test, y_test = df_to_XY(test_subset, features)  # Prepare testing data

            if len(y_test) == 0:
                continue  # Skip if the testing data is empty for the period

            predictions = model.predict(X_test)  # Predict on the testing data
            results.append((train_period, test_period, y_test, predictions))  # Store results

    if save:
        with open(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/results/{name}.pkl', 'wb') as f:  # Save results to a pickle file
            pickle.dump(results, f)

    return results

# Calculate SHAP values for feature importance
def calculate_shap_values(df_train, df_test, features, save=True, name=None):
    if name is None:
        name = 'shap_values'

    shap_values_list = []
    test_list = []
    year_month_list = df_train['YearMonth'].unique()  # Get unique year-month periods

    for year_month in year_month_list:
        X_train, y_train = df_to_XY(df_train[df_train['YearMonth'] == year_month], features)  # Prepare training data
        model = train_model(X_train, y_train)  # Train the model
        explainer = shap.TreeExplainer(model)  # Initialize SHAP explainer

        X_test, y_test = df_to_XY(df_test[df_test['YearMonth'] == year_month], features)  # Prepare testing data

        if len(y_test) == 0:
            continue  # Skip if the testing data is empty for the period

        shap_values = explainer.shap_values(X_test)  # Calculate SHAP values

        test_list.append(pd.DataFrame(X_test, columns=features))  # Store the test data
        shap_values_list.append(shap_values)  # Store the SHAP values

    if save:
        with open(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/results/{name}.pkl', 'wb') as f:
            pickle.dump((test_list, shap_values_list), f)
    return test_list, shap_values_list

# Functions to run the tests for different feature sets
def run_feature_set(name, features, train, test):
    simultaneous_test(train, test, features, name=f'{name}_simultaneous')
    nonsimultaneous_test(train, test, features, name=f'{name}_nonsimultaneous')
    calculate_shap_values(train, test, features, name=f'{name}_SHAP')

# Run classification for all feature sets on the given data path
def run_classification(data_path, feature_sets):
    data = pd.read_csv(data_path)
    train, test = get_edge_slice(data)

    for feature_set_name, features in feature_sets.items():
        run_feature_set(f'rail_classification_{feature_set_name}', features, train, test)

if __name__ == "__main__":
    feature_sets = {
        'TF': ['CN', 'SA', 'JA', 'SO', 'HPI', 'HDI', 'LHNI', 'PA', 'AA', 'RA', 'LPI'],
        'WTF': ['weighted_CN', 'weighted_SA', 'weighted_JA', 'weighted_SO', 'weighted_HPI', 'weighted_HDI', 'weighted_LHNI', 'weighted_PA', 'weighted_AA', 'weighted_RA', 'weighted_LPI'],
        'Weight': ['Rides planned'],
        'Distance': ['distance']
    }

    data_path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc-Thesis-main/adapted/data/features/monthly_features_per_trajectory.csv' 
    run_classification(data_path, feature_sets)
