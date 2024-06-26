import sys
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load data
data_path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/features/monthly_features_per_trajectory.csv'
data = pd.read_csv(data_path)

# Time-based split function to ensure each month has data in both sets
def time_based_split(data, train_ratio=0.7):
    train_data = []
    test_data = []

    for period in data['YearMonth'].unique():
        period_data = data[data['YearMonth'] == period]
        train_sample_count = int(len(period_data) * train_ratio)
        
        period_train = period_data.sample(train_sample_count, random_state=42)
        period_test = period_data.drop(period_train.index)
        
        train_data.append(period_train)
        test_data.append(period_test)

    df_train = pd.concat(train_data)
    df_test = pd.concat(test_data)

    return df_train, df_test

train, test = time_based_split(data)

# Define the feature sets
feature_sets = {
    'TF': ['CN', 'SA', 'JA', 'SO', 'HPI', 'HDI', 'LHNI', 'PA', 'AA', 'RA', 'LPI'],
    'WTF': ['weighted_CN', 'weighted_SA', 'weighted_JA', 'weighted_SO', 'weighted_HPI', 'weighted_HDI', 'weighted_LHNI', 'weighted_PA', 'weighted_AA', 'weighted_RA', 'weighted_LPI'],
    'NCM': ['source_closeness', 'target_closeness', 'source_degree', 'target_degree', 'source_strength', 'target_strength']
}

# Define model training function
def train_model(X_train, y_train, classifier, param_grid=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    if param_grid:
        randomized_search = RandomizedSearchCV(
            classifier, 
            param_distributions=param_grid, 
            n_iter=10, 
            scoring='balanced_accuracy', 
            n_jobs=-1, 
            cv=10, 
            random_state=42)
        randomized_search.fit(X_train, y_train)
        best_classifier = randomized_search.best_estimator_
    else:
        classifier.fit(X_train, y_train)
        best_classifier = classifier
        
    return best_classifier, scaler

# Define evaluation function
def evaluate_classifier(y_true, y_pred, y_proba=None):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan, np.nan, np.nan  # Return NaN if arrays are empty

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if y_proba is not None and len(y_proba) == len(y_true):
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = np.nan  # Return NaN if probabilities are not available or not the same length
    return balanced_accuracy, f1, auc

# Define function to convert dataframe to X, y
def df_to_XY(df, features, target='Significant Delay'):
    X = df[features].to_numpy()
    y = df[target].astype(int).to_numpy()
    return X, y

# Define Simultaneous Testing Function
def simultaneous_test(df_train, df_test, features, classifier, param_grid=None, name=None):
    year_month_list = df_train['YearMonth'].unique()
    res_df_test = df_test.copy()
    res_df_test[f'{name}_pred'] = np.nan
    res_df_test[f'{name}_proba'] = np.nan

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for year_month in tqdm(year_month_list):
        train_subset = df_train[df_train['YearMonth'] == year_month]
        test_subset = df_test[df_test['YearMonth'] == year_month]

        X_train, y_train = df_to_XY(train_subset, features)
        X_test, y_test = df_to_XY(test_subset, features)

        if len(X_train) == 0 or len(X_test) == 0:
            logging.warning(f"Skipping YearMonth {year_month} due to empty training or testing data.")
            continue

        model, scaler = train_model(X_train, y_train, classifier, param_grid)
        X_test = scaler.transform(X_test)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        res_df_test.loc[res_df_test['YearMonth'] == year_month, f'{name}_pred'] = y_pred
        if y_proba is not None:
            res_df_test.loc[res_df_test['YearMonth'] == year_month, f'{name}_proba'] = y_proba

        # Store intermediate results
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            all_y_proba.extend(y_proba)

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        if all_y_proba:
            balanced_accuracy, f1, auc = evaluate_classifier(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba))
        else:
            balanced_accuracy, f1, auc = evaluate_classifier(np.array(all_y_true), np.array(all_y_pred))
    else:
        balanced_accuracy, f1, auc = np.nan, np.nan, np.nan

    print(f"{name} - Balanced Accuracy: {balanced_accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    res_df_test.to_csv(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results_classifiers/{name}_results.csv', index=False)
    return res_df_test, all_y_true, all_y_pred, all_y_proba

# Define Non-Simultaneous Testing Function
def nonsimultaneous_test(df_train, df_test, features, classifier, param_grid=None, name=None):
    year_month_list = df_train['YearMonth'].unique()
    results = []

    for i, train_period in enumerate(tqdm(year_month_list)):
        train_subset = df_train[df_train['YearMonth'] == train_period]
        X_train, y_train = df_to_XY(train_subset, features)

        if len(y_train) == 0:
            logging.warning(f"Skipping train_period {train_period} due to empty training data.")
            continue

        model, scaler = train_model(X_train, y_train, classifier, param_grid)

        for test_period in year_month_list[i:]:
            test_subset = df_test[df_test['YearMonth'] == test_period]
            X_test, y_test = df_to_XY(test_subset, features)

            if len(y_test) == 0:
                logging.warning(f"Skipping test_period {test_period} due to empty testing data.")
                continue

            X_test = scaler.transform(X_test)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            results.append((train_period, test_period, y_test.tolist(), y_pred.tolist(), y_proba.tolist() if y_proba is not None else []))

    results_df = pd.DataFrame(results, columns=['TrainPeriod', 'TestPeriod', 'TrueLabels', 'Predictions', 'Probabilities'])
    results_df.to_csv(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results_classifiers/{name}_nonsimultaneous_results.csv', index=False)

    all_y_true = [label for result in results for label in result[2]]
    all_y_pred = [pred for result in results for pred in result[3]]
    all_y_proba = [proba for result in results for proba in result[4]]

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        if all_y_proba:
            balanced_accuracy, f1, auc = evaluate_classifier(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba))
        else:
            balanced_accuracy, f1, auc = evaluate_classifier(np.array(all_y_true), np.array(all_y_pred))
    else:
        balanced_accuracy, f1, auc = np.nan, np.nan, np.nan

    print(f"{name} - Balanced Accuracy: {balanced_accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    return results_df, all_y_true, all_y_pred, all_y_proba

# Define parameter grids for each classifier
param_grids = {
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    },
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 500, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'DecisionTreeClassifier': {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
}

# Specific parameters for XGBoost
best_params = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
           'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
           'n_estimators': 625, 'objective':'binary:logistic', 'max_depth': 5, 'min_child_weight': 6}

# Include classifiers dictionary with XGBoost
classifiers = {
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'XGBClassifier': XGBClassifier(**best_params),
}

# Run simultaneous and non-simultaneous testing for each classifier and each feature set
for feature_set_name, feature_set in feature_sets.items():
    for name, classifier in classifiers.items():
        param_grid = param_grids.get(name, None)
        print(f"Running simultaneous test for {name} with feature set {feature_set_name}...")
        simultaneous_test(train, test, feature_set, classifier, param_grid, f'{name}_{feature_set_name}')
        
        print(f"Running non-simultaneous test for {name} with feature set {feature_set_name}...")
        nonsimultaneous_test(train, test, feature_set, classifier, param_grid, f'{name}_{feature_set_name}')