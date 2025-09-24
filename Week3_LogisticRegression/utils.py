import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np

'''Function to load the dataset'''
def load_dataset(data_file):
    current_path = os.getcwd() # get the current working directory 
    data_path = os.path.join(current_path, 'data', data_file) # join the path to get the absolute file_path
    df = pd.read_csv(data_path) # load the data using pandas
    return df

'''Function to inspect the dataset'''
def data_inspection(dataset):
    print('---- DATA INFORMATION ----')
    print(dataset.info())
    print()
    print('---- DATA DESCRIPTION ----')
    print(dataset.describe())
    print()
    print('---- DATA SHAPE ----')
    print(dataset.shape)
    print()
    print('---- SAMPLE DATA ----')
    print(dataset.head())

'''Function to plot the heatmap'''
def plot_heatmap(dataframe):
    plt.figure(figsize=(8,6))
    sns.heatmap(dataframe.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Standardized Features")
    plt.show()

'''Function to plot the pairplot'''
def plot_pairplot(dataframe, features, hue=None, diag_kind='hist', palette='Set2'):
    """
    Plots a seaborn pairplot for selected features.
    """
    if hue:
        sns.pairplot(
            dataframe[features],
            hue=hue,
            diag_kind=diag_kind,
            palette=palette
        )
    else:
        sns.pairplot(
            dataframe[features],
            diag_kind=diag_kind
        )
    plt.show()

'''Function to evaluate the metrics for single linear regression model'''
def evaluate_metrics_single_lr(models, training_features, X_train, X_test, y_train, y_test):
    """
    Evaluate multiple models on their respective features and display metrics in a table.
    
    Parameters:
        models: list of dicts, each containing {'model': trained_model}
        training_features: list of feature names, same order as models
        X_train, X_test: pd.DataFrame containing training and test features
        y_train, y_test: pd.Series or np.array containing target values
    """
    results = []

    for model, feature_name in zip(models, training_features):
        # Extract feature column
        X_tr = X_train[[feature_name]].to_numpy()
        X_te = X_test[[feature_name]].to_numpy()
        
        # Predictions
        y_pred_train = model['model'].predict(X_tr).flatten()
        y_pred_test = model['model'].predict(X_te).flatten()
        
        # Calculate metrics
        metrics = {
            'Feature': feature_name,
            'Train RMSE': root_mean_squared_error(y_train, y_pred_train),
            'Test RMSE': root_mean_squared_error(y_test, y_pred_test),
            'Train R2': r2_score(y_train, y_pred_train),
            'Test R2': r2_score(y_test, y_pred_test),
            'Train MAE': mean_absolute_error(y_train, y_pred_train),
            'Test MAE': mean_absolute_error(y_test, y_pred_test)
        }
        results.append(metrics)
    
    # Convert to DataFrame for nice table display
    results_df = pd.DataFrame(results)
    
    # format floats for better readability
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    return results_df

'''Function to evaluate the metrics for multiple linear regression model'''
def evaluate_metrics_multiple_lr(model, X_train, X_test, y_train, y_test):
    results = []
    
    X_train_temp = X_train.to_numpy().astype(np.float32)
    X_test_temp = X_test.to_numpy().astype(np.float32)
    
    y_pred_train = model['model'].predict(X_train_temp).flatten()
    y_pred_test = model['model'].predict(X_test_temp).flatten()
                       
    metrics = {
            'Train RMSE': root_mean_squared_error(y_train, y_pred_train),
            'Test RMSE': root_mean_squared_error(y_test, y_pred_test),
            'Train R2': r2_score(y_train, y_pred_train),
            'Test R2': r2_score(y_test, y_pred_test),
            'Train MAE': mean_absolute_error(y_train, y_pred_train),
            'Test MAE': mean_absolute_error(y_test, y_pred_test)
        }
    results.append(metrics)

    # Convert to DataFrame for nice table display
    results_df = pd.DataFrame(results)

    # format floats for better readability
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    return results_df