import json
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import src.utils as ut
from src.config import scoring_methods, scoring_mode, paths
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

def generate_classification_report(y_test, y_pred):
    """
    Generates a classification report in DataFrame format
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred (pd.Series): Predicted values
    Returns:
        pd.DataFrame: Classification report 
    """
    report = classification_report(y_test, y_pred, output_dict = True)
    report_df = pd.DataFrame(report).transpose()
    
    accuracy = report_df.loc['accuracy', 'precision']
    report_df = report_df.drop(index = 'accuracy')
    
    print('Classification report')
    display(report_df)
    
    print(f'Test Accuracy: {accuracy:.2%}')
   
    # document a log
    ut.write_log(f'classification report generated')
    

def plot_confusion_matrix(y_test, y_pred, labels = None):
    """
    Shows a plot of the confusion matrix plot
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred (pd.Series): Predicted values
        - labels (list, optional): Class labels
    """
    # if labels not specified, generate automatically
    if labels is None:
        labels = sorted(set(y_test).union(set(y_pred)))

    conf_matrix = confusion_matrix(y_test, y_pred, labels = labels)
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = labels, yticklabels = labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # document a log
    ut.write_log(f'confusion matrix displayed')

def plot_roc_curve(y_test, y_pred_proba):
    """
    Shows ROC curve and calculates AUC
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred_proba (np.array): Probability predictions
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize = (10, 8))
    plt.plot(fpr, tpr, label = f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()
    
    # document a log
    ut.write_log(f'roc curve displayed')
    
def plot_precision_recall(y_test, y_pred_proba):
    """
    Plot a curve of precision vs recall
    
    Parameters:
        - y_test (pd.Series): Real values of target variable
        - y_pred_proba (np.array): Probability predictions
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

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
        json.dump(metrics, file, indent = 4)
    
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
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # main metrics
    metrics = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred, average = 'binary', zero_division = 0),
        'recall' : recall_score(y_test, y_pred, average = 'binary', zero_division = 0),
        'f1_score' : f1_score(y_test, y_pred, average = 'binary')
    }
    
    # check if AUC-ROC is available
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return metrics