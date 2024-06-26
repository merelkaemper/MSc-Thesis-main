import sys
import random
import datetime
import shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

from datetime import date
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, mean_squared_error, r2_score, mean_absolute_error

sys.path.append('src')
from plot_style import *

def get_trajectory_slice(data, f_train_e=0.7, seed=30):
    df = data
    trajectories = list(set(df[['source', 'target']].itertuples(index=False, name=None)))
    random.seed(seed)
    train_size = int(f_train_e * len(trajectories))
    random.shuffle(trajectories)
    train_trajectories = set(trajectories[:train_size])
    test_trajectories = set(trajectories[train_size:])
    
    df_train = df.loc[df[['source', 'target']].apply(tuple, axis=1).isin(train_trajectories)].copy()
    df_test = df.loc[df[['source', 'target']].apply(tuple, axis=1).isin(test_trajectories)].copy()
    
    return df_train, df_test

def df_to_XY(df, features, target='Significant Delay'):
    X = df.loc[:, features].to_numpy()
    y = df.loc[:, target].to_numpy()
    return X, y

def simultaneous_test(df_train, df_test, features, best_params, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_simultaneous'
    else:
        name = name + '_simultaneous'
    
    year_month_list = sorted(df_train['YearMonth'].unique())
    res_df_test = df_test.copy()
    res_df_test['simultaneous_pred'] = np.nan
    res_df_test['simultaneous_null'] = np.nan
    
    for year_month in tqdm(year_month_list):
        X_train, y_train = df_to_XY(df_train[df_train['YearMonth'] == year_month], features)
        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_test, y_test = df_to_XY(df_test[df_test['YearMonth'] == year_month], features)
        
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        model_null = XGBClassifier(**best_params)
        model_null.fit(X_train, y_train_null)
        
        y_pred = model.predict(X_test)
        y_pred_null = model_null.predict(X_test)
        
        res_df_test.loc[res_df_test['YearMonth'] == year_month, 'simultaneous_pred'] = y_pred
        res_df_test.loc[res_df_test['YearMonth'] == year_month, 'simultaneous_null'] = y_pred_null
    
    if save:
        res_df_test.to_csv(f'./results_XGB/{name}.csv', index=False)
    
    return res_df_test

def nonsimultaneous_test(df_train, df_test, features, best_params, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_nonsimultaneous'
    else:
        name = name + '_nonsimultaneous'
    
    year_month_list = sorted(df_test['YearMonth'].unique())
    preds = []
    
    for year_month_train in tqdm(year_month_list):
        X_train, y_train = df_to_XY(df_train[df_train['YearMonth'] == year_month_train], features)
        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
        y_train_null = y_train.copy()
        np.random.shuffle(y_train_null)
        
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        model_null = XGBClassifier(**best_params)
        model_null.fit(X_train, y_train_null)
        
        for year_month_test in year_month_list:
            if year_month_test < year_month_train:
                continue
            X_test, y_test = df_to_XY(df_test[df_test['YearMonth'] == year_month_test], features)
            y_pred = model.predict(X_test)
            y_null = model_null.predict(X_test)
            preds.append([year_month_train, year_month_test, y_test, y_pred, y_null])
    
    if save:
        import pickle
        with open(f'./results_XGB/{name}.pkl', 'wb') as f:
            pickle.dump(preds, f)
    
    return preds

def all_shap_values(df_train, df_test, features, best_params, save=True, name=None):
    if name is None:
        name = ''.join([w[0] for w in features]) + '_SHAP'
    else:
        name = name + '_SHAP'
    
    def get_temporal_order(shap_list):
        importance_array = []
        for shap_values in shap_list:
            array = -np.abs(shap_values).mean(axis=0)
            ranks = ss.rankdata(array)
            importance_array.append(ranks)
        return np.array(importance_array)

    shap_values_list = []
    test_list = []
    year_month_list = sorted(df_test['YearMonth'].unique())
    
    for year_month in tqdm(year_month_list):
        X_train, y_train = df_to_XY(df_train[df_train['YearMonth'] == year_month], features)
        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_test, y_test = df_to_XY(df_test[df_test['YearMonth'] == year_month], features)
        
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        test_list.append(pd.DataFrame(X_test, columns=features))
        shap_values_list.append(shap_values)
    
    if save:
        import pickle
        with open(f'./results_XGB/{name}.pkl', 'wb') as f:
            pickle.dump((test_list, year_month_list, shap_values_list), f)
    
    return test_list, year_month_list, shap_values_list

def BTF(train, test):
    name = 'NS_Classification_BTF'
    features = ['CN', 'SA', 'JA', 'SO', 'HPI', 'HDI', 'LHNI', 'PA', 'AA', 'RA', 'LPI']
    simultaneous_test(train, test, features, best_params, name=name)
    nonsimultaneous_test(train, test, features, best_params, name=name)
    all_shap_values(train, test, features, best_params, name=name)

def WTF(train, test):
    name = 'NS_Classification_WTF'
    features = ['weighted_CN', 'weighted_SA', 'weighted_JA', 'weighted_SO', 'weighted_HPI', 
                'weighted_HDI', 'weighted_LHNI', 'weighted_PA', 'weighted_AA', 'weighted_RA', 'weighted_LPI']
    simultaneous_test(train, test, features, best_params, name=name)
    nonsimultaneous_test(train, test, features, best_params, name=name)
    all_shap_values(train, test, features, best_params, name=name)

def NCM(train, test):
    name = 'NS_Classification_NCM'
    features = ['source_closeness', 'target_closeness', 'source_degree', 'target_degree', 'source_strength', 'target_strength']
    simultaneous_test(train, test, features, best_params, name=name)
    nonsimultaneous_test(train, test, features, best_params, name=name)
    all_shap_values(train, test, features, best_params, name=name)

if __name__ == "__main__":
    global best_params
    best_params = {'lambda': 0.5650701862593042, 'alpha': 0.0016650896783581535,
                   'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.009,
                   'n_estimators': 625, 'objective': 'binary:logistic', 'max_depth': 5, 'min_child_weight': 6}

    data_path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/features/monthly_features_per_trajectory.csv'
    data = pd.read_csv(data_path)
    data['YearMonth'] = pd.to_datetime(data['YearMonth']).dt.to_period('M')
    
    train, test = get_trajectory_slice(data)
    BTF(train, test)
    WTF(train, test)
    NCM(train, test)














# import sys
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
# import matplotlib.pyplot as plt
# import logging

# # Define classifiers
# classifiers = {
#     'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
#     'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
#     'RidgeClassifier': RidgeClassifier(),
#     'RidgeClassifierCV': RidgeClassifierCV(),
#     'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
#     'LogisticRegressionCV': LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
#     'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
#     'XGBClassifier': XGBClassifier(random_state=42),
#     'DutchDraw': None  # Placeholder for DutchDraw, will handle separately
# }

# # Hyperparameter grids for RandomizedSearchCV
# param_grids = {
#     'GradientBoostingClassifier': {
#         'n_estimators': [100, 200, 300],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 4, 5]
#     },
#     'AdaBoostClassifier': {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 1]
#     },
#     'RidgeClassifier': {
#         'alpha': [0.1, 1, 10]
#     },
#     'RidgeClassifierCV': {
#         'alphas': [[0.1, 1, 10]]
#     },
#     'LogisticRegression': {
#         'C': [0.1, 1, 10]
#     },
#     'LogisticRegressionCV': {
#         'Cs': [0.1, 1, 10]
#     },
#     'LinearDiscriminantAnalysis': {
#         'solver': ['svd', 'lsqr', 'eigen']
#     },
#     'XGBClassifier': {
#         'n_estimators': [100, 200, 300],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 4, 5]
#     }
# }

# # Load data
# data_path = '/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/features/monthly_features_per_trajectory.csv'
# data = pd.read_csv(data_path)

# # Time-based split function to ensure each month has data in both sets
# def time_based_split(data, train_ratio=0.7):
#     train_data = []
#     test_data = []

#     for period in data['YearMonth'].unique():
#         period_data = data[data['YearMonth'] == period]
#         train_sample_count = int(len(period_data) * train_ratio)
        
#         period_train = period_data.sample(train_sample_count, random_state=42)
#         period_test = period_data.drop(period_train.index)
        
#         train_data.append(period_train)
#         test_data.append(period_test)

#     df_train = pd.concat(train_data)
#     df_test = pd.concat(test_data)

#     return df_train, df_test

# def df_to_XY(df, features, target='Significant Delay'):
#     X = df[features].to_numpy()
#     y = df[target].astype(int).to_numpy()
#     return X, y

# # Define the feature sets
# feature_sets = {
#     'TF': ['CN', 'SA', 'JA', 'SO', 'HPI', 'HDI', 'LHNI', 'PA', 'AA', 'RA', 'LPI'],
#     'WTF': ['weighted_CN', 'weighted_SA', 'weighted_JA', 'weighted_SO', 'weighted_HPI', 'weighted_HDI', 'weighted_LHNI', 'weighted_PA', 'weighted_AA', 'weighted_RA', 'weighted_LPI'],
#     'Demographic and Spatial': ['distance', 'population_source', 'population_target', 'gravitational_index'],
#     'Centrality measures': ['source_betweenness', 'target_betweenness', 'source_closeness', 'target_closeness', 'source_degree', 'target_degree']
# }

# def train_model(X_train, y_train, classifier, param_grid):
#     # Check the number of samples in the minority class
#     minority_class_size = sum(y_train == 1)  # Assuming 1 is the minority class
#     n_neighbors = min(5, minority_class_size - 1)  # Adjust n_neighbors based on the minority class size
    
#     if minority_class_size >= 2:  # Ensure at least two samples to apply SMOTE
#         smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=n_neighbors)
#         X_train, y_train = smote.fit_resample(X_train, y_train)
#     else:
#         logging.warning(f"Not enough minority class samples for SMOTE: {minority_class_size} samples")

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
    
#     if param_grid:
#         randomized_search = RandomizedSearchCV(classifier, param_distributions=param_grid, n_iter=10, scoring='balanced_accuracy', n_jobs=-1, cv=3, random_state=42)
#         randomized_search.fit(X_train, y_train)
#         best_classifier = randomized_search.best_estimator_
#     else:
#         classifier.fit(X_train, y_train)
#         best_classifier = classifier
        
#     return best_classifier, scaler

# # Define evaluation function
# def evaluate_classifier(y_true, y_pred):
#     balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     auc = roc_auc_score(y_true, y_pred)
#     return balanced_accuracy, f1, auc

# # Define simultaneous test
# def simultaneous_test(df_train, df_test, features, classifier, param_grid, name=None):
#     year_month_list = df_train['YearMonth'].unique()
#     res_df_test = df_test.copy()
#     res_df_test[f'{name}_pred'] = np.nan

#     for year_month in tqdm(year_month_list):
#         train_subset = df_train[df_train['YearMonth'] == year_month]
#         test_subset = df_test[df_test['YearMonth'] == year_month]

#         X_train, y_train = df_to_XY(train_subset, features)
#         X_test, y_test = df_to_XY(test_subset, features)

#         print(f"YearMonth: {year_month}, X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")

#         if len(X_train) == 0 or len(X_test) == 0:
#             logging.warning(f"Skipping YearMonth {year_month} due to empty training or testing data.")
#             continue

#         model, scaler = train_model(X_train, y_train, classifier, param_grid)
#         X_test = scaler.transform(X_test)

#         y_pred = model.predict(X_test)
#         res_df_test.loc[res_df_test['YearMonth'] == year_month, f'{name}_pred'] = y_pred

#     res_df_test.to_csv(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results/{name}_results.csv', index=False)
#     return res_df_test

# # Define non-simultaneous test
# def nonsimultaneous_test(df_train, df_test, features, classifier, param_grid, name=None):
#     year_month_list = df_train['YearMonth'].unique()
#     results = []

#     for i, train_period in enumerate(tqdm(year_month_list)):
#         train_subset = df_train[df_train['YearMonth'] == train_period]
#         X_train, y_train = df_to_XY(train_subset, features)

#         if len(X_train) == 0:
#             logging.warning(f"Skipping train_period {train_period} due to empty training data.")
#             continue

#         model, scaler = train_model(X_train, y_train, classifier, param_grid)

#         for test_period in year_month_list[i:]:
#             test_subset = df_test[df_test['YearMonth'] == test_period]
#             X_test, y_test = df_to_XY(test_subset, features)

#             if len(X_test) == 0:
#                 logging.warning(f"Skipping test_period {test_period} due to empty testing data.")
#                 continue

#             X_test = scaler.transform(X_test)
#             y_pred = model.predict(X_test)
#             results.append((train_period, test_period, y_test.tolist(), y_pred.tolist()))

#     results_df = pd.DataFrame(results, columns=['TrainPeriod', 'TestPeriod', 'TrueLabels', 'Predictions'])
#     results_df.to_csv(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results/{name}_nonsimultaneous_results.csv', index=False)
#     return results_df

# # Define DutchDraw test
# def dutchdraw_test(df_train, df_test, features, name=None):
#     year_month_list = df_train['YearMonth'].unique()
#     res_df_test = df_test.copy()
#     res_df_test[f'{name}_pred'] = np.nan

#     for year_month in tqdm(year_month_list):
#         test_subset = df_test[df_test['YearMonth'] == year_month]

#         X_test, y_test = df_to_XY(test_subset, features)

#         if len(y_test) == 0:
#             logging.warning(f"Skipping YearMonth {year_month} due to empty testing data.")
#             continue
        
#         # Obtain DutchDraw baseline predictions
#         baseline = dutchdraw.baseline_functions(y_true=y_test, measure='ACC')
#         y_pred = baseline['Fast Expectation Function'](theta=0.5)  # Assuming a balanced scenario

#         res_df_test.loc[res_df_test['YearMonth'] == year_month, f'{name}_pred'] = y_pred

#     res_df_test.to_csv(f'/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results/{name}_results.csv', index=False)
#     return res_df_test

# # Collect Results
# results_simultaneous = []
# results_nonsimultaneous = []

# train, test = time_based_split(data)

# # Debugging: Ensure no empty arrays in training or testing sets
# for period in sorted(train['YearMonth'].unique()):
#     print(f"Train period {period}: {len(train[train['YearMonth'] == period])} samples")
# for period in sorted(test['YearMonth'].unique()):
#     print(f"Test period {period}: {len(test[test['YearMonth'] == period])} samples")

# for feature_set_name, features in feature_sets.items():
#     for clf_name, clf in classifiers.items():
#         param_grid = param_grids.get(clf_name, None)

#         if clf_name == 'DutchDraw':
#             print(f"Running {clf_name} on feature set {feature_set_name} for simultaneous testing")
#             res_df = dutchdraw_test(train, test, features, name=f'{clf_name}_{feature_set_name}_simultaneous')
#             y_true = res_df['Significant Delay']
#             y_pred = res_df[f'{clf_name}_{feature_set_name}_simultaneous_pred']
#             balanced_accuracy, f1, auc = evaluate_classifier(y_true, y_pred)
#             results_simultaneous.append({'Classifier': clf_name, 'Feature Set': feature_set_name, 'Balanced Accuracy': balanced_accuracy, 'F1 Score': f1, 'AUC': auc})

#             print(f"Running {clf_name} on feature set {feature_set_name} for non-simultaneous testing")
#             res_df = dutchdraw_test(train, test, features, name=f'{clf_name}_{feature_set_name}')
#             y_true = np.concatenate(res_df['TrueLabels'].values)
#             y_pred = np.concatenate(res_df['Predictions'].values)
#             balanced_accuracy, f1, auc = evaluate_classifier(y_true, y_pred)
#             results_nonsimultaneous.append({'Classifier': clf_name, 'Feature Set': feature_set_name, 'Balanced Accuracy': balanced_accuracy, 'F1 Score': f1, 'AUC': auc})
#         else:
#             print(f"Training {clf_name} on feature set {feature_set_name} for simultaneous testing")
#             res_df = simultaneous_test(train, test, features, clf, param_grid, name=f'{clf_name}_{feature_set_name}_simultaneous')
#             y_true = res_df['Significant Delay']
#             y_pred = res_df[f'{clf_name}_{feature_set_name}_simultaneous_pred']
#             balanced_accuracy, f1, auc = evaluate_classifier(y_true, y_pred)
#             results_simultaneous.append({'Classifier': clf_name, 'Feature Set': feature_set_name, 'Balanced Accuracy': balanced_accuracy, 'F1 Score': f1, 'AUC': auc})

#             print(f"Training {clf_name} on feature set {feature_set_name} for non-simultaneous testing")
#             res_df = nonsimultaneous_test(train, test, features, clf, param_grid, name=f'{clf_name}_{feature_set_name}')
#             y_true = np.concatenate(res_df['TrueLabels'].values)
#             y_pred = np.concatenate(res_df['Predictions'].values)
#             balanced_accuracy, f1, auc = evaluate_classifier(y_true, y_pred)
#             results_nonsimultaneous.append({'Classifier': clf_name, 'Feature Set': feature_set_name, 'Balanced Accuracy': balanced_accuracy, 'F1 Score': f1, 'AUC': auc})

# # Save results to CSV
# results_simultaneous_df = pd.DataFrame(results_simultaneous)
# results_simultaneous_df.to_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results/simultaneous_results.csv', index=False)

# results_nonsimultaneous_df = pd.DataFrame(results_nonsimultaneous)
# results_nonsimultaneous_df.to_csv('/Users/merelkamper/Documents/MSc Data Science/Thesis/MSc_thesis_code/results/nonsimultaneous_results.csv', index=False)
