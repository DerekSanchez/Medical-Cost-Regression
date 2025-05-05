import json
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import src.utils as ut
from src.config import scoring_methods, scoring_mode, paths
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

def generate_regression_report(y_test, y_pred):
    """
    Generates a regression report in DataFrame format
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred (pd.Series): Predicted values
    Returns:
        pd.DataFrame: Regression report 
    """
    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (RMSE)': root_mean_squared_error(y_test, y_pred),
        'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(y_test, y_pred),
        'R-squared (R2)': r2_score(y_test, y_pred)
    }
    
    report_df = pd.DataFrame(metrics, index=['Value']).transpose()
    
    print('Regression report')
    display(report_df)
    
    # document a log
    ut.write_log(f'regression report generated')

def plot_residuals(y_test, y_pred):
    """
    Shows a plot of residuals
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred (pd.Series): Predicted values
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
    
    # document a log
    ut.write_log(f'residuals plot displayed')

def plot_predictions(y_test, y_pred):
    """
    Shows a scatter plot of actual vs predicted values
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred (pd.Series): Predicted values
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.show()
    
    # document a log
    ut.write_log(f'actual vs predicted plot displayed')

def save_metrics(metrics, model_name):
    """
    Saves testing metrics on a JSON file
    
    Parameters:
        - metrics (dict): Testing results
        - model_name (str): Name of evaluated model
    """
    # path to save metrics
    metrics_path = f"{paths['models']}/{model_name}_metrics.json"
    
    with open(metrics_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    
    print(f'Metrics saved on {metrics_path}')

def get_test_metrics(model, X_test, y_test):
    """
    Calculates relevant metrics for a model on the test set
    
    Parameters:
        - model: Trained model
        - X_test (pd.DataFrame): Test set features
        - y_test (pd.Series): Test set target
        
    Returns:
        - dict: Dictionary with calculated metrics
    """
    # predictions
    y_pred = model.predict(X_test)
    
    # main metrics
    metrics = {
        'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        'mean_squared_error': mean_squared_error(y_test, y_pred),
        'root_mean_squared_error': mean_squared_error(y_test, y_pred, squared=False),
        'mean_absolute_percentage_error': mean_absolute_percentage_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    
    return metrics
