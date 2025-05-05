# ===============================
# Libraries and Configuration
# ===============================

# -------------------------------
# Path Setting
# -------------------------------

import sys
from pathlib import Path

project_root = Path().resolve().parent # set path to project root
sys.path.append(str(project_root))

# -------------------------------
# Libraries and Dependencies
# -------------------------------

import pandas as pd
import numpy as np
import src.config as cf
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, RandomOverSampler


# -------------------------------
# Raise Error for Necessary Columns for Prep Step
# -------------------------------

class MissingColumnError(Exception):
    """
    Personalized exception for necessary columns in each step in preprocesssing.
    Raises an error if the column is not in the input data
    """
    def __init__(self, missing_columns):
        self.message = f"The following missing columns are missing: {', '.join(missing_columns)}"
        super().__init__(self.message)

# ===============================
# Target Variable Preprocessing
# ===============================

# No target preprocessing needed for this dataset        

# ===============================
# Feature Selection
# ===============================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects and drops irrelevant columns based on EDA analysis
    """
    
    def __init__(self):
        # optional attributes to store dropped columns
        self.dropped_features = None
    
    def fit(self, X, y = None):
        """
        Adjusts the class identifying the features to drop
        """
        # append lists of numerical and categorical features to drop
        self.dropped_features = cf.cat_features_to_drop + cf.num_features_to_drop
        
        return self
    
    def transform(self, X):
        """
        Drop irrelevant features from dataset
        """
        X_copy = X.copy()
        
        # verify if needed columns exist in data
        required_columns = self.dropped_features
        missing_columns = [col for col in required_columns if col not in X_copy.columns]

        if missing_columns:
            raise MissingColumnError(missing_columns)
        
        X_copy = X_copy.drop(columns = self.dropped_features, errors = 'ignore').reset_index()
        
        return X_copy

# ===============================
# Data Cleaning
# ===============================

class DataCleaning(BaseEstimator, TransformerMixin):
    """
    Data Cleaning Class:
        - Data type adjustment
        - Handling inconsisting labels
        - Stadardize columns in inconsistent formatting
    """
    
    def __init__(self):
        pass
    
    def fit(self, X):
        """
        Fit Data Cleaning computations
        """
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        # adjust spaces in string columns
        object_columns = X_copy.select_dtypes(include = 'object').columns
        for col in object_columns:
            X_copy[col] = X_copy[col].str.strip()
    
        return X_copy


# ===============================
# Feature Engineering
# ===============================

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Feature Engineering including:
        - Creation of new columns
        - Transformation of existing columns
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        """
        Fits any calculation to perform Feature Engineering
        """
        
        return self
    
    def transform(self, X):
        """
        Apply transformations and create new features
        """ 
        X_copy = X.copy()
        
        # create new column: age squared
        if 'age' in X_copy.columns:
            X_copy['age2'] = X_copy['age'] ** 2
        
        # create new column: log bmi
        if 'bmi' in X_copy.columns:
            X_copy['Logbmi'] = X_copy['bmi'].apply(
            lambda x: np.log(x) if pd.notnull(x) and x > 0 else 0)
        
        return X_copy

# ===============================
# Outlier Detection
# ===============================

class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Outlier detection based on Inter Quartile Range (IQR)
    - Identifies values outside the range defined by the threshold
    - allows the user to identify or eliminate outliers
    """
    
    def __init__(self, multiplier = 1.5 , action = 'cap'):
        """
        Parameters:
            multiplier (float): multiplier for IQR. default is 1.5
            action (str): defines what to do with the outlier:
                - 'remove' to delete them
                - 'cap' to cap them
                - 'flag' to flag them  
        """
        self.multiplier = multiplier
        self.action = action

    def fit(self, X, y = None):
        """
        Fits necessary statistics (mean and sd)
        """
        numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns
        self.q1 = X[numerical_cols].quantile(0.25)
        self.q3 = X[numerical_cols].quantile(0.75)
        self.iqr = self.q3 - self.q1 
        
        # compute limits
        self.lower_bound = self.q1 - self.multiplier * self.iqr
        self.upper_bound = self.q3 + self.multiplier * self.iqr
        
        return self
    
    def transform(self, X):
        """
        Applies outlier detection and handling as specified
        """
        X_copy = X.copy()
        
        # select numerical columns
        numeric_cols = X_copy.select_dtypes(include = ['int64', 'float64']).columns
        
        # raise error if not in the dataset
        required_columns = numeric_cols
        missing_columns = [col for col in required_columns if col not in X_copy.columns]
        
        if missing_columns:
            raise MissingColumnError(missing_columns)
        
        # identify outliers
        outliers = (X_copy[numeric_cols] < self.lower_bound) | (X_copy[numeric_cols] > self.upper_bound)
        
        if self.action == 'remove':
            # remove rows with outliers
            X_copy = X_copy[~outliers.any(axis = 1)]
            
        elif self.action == 'cap':
            # cap values
            X_copy[numeric_cols] = X_copy[numeric_cols].clip(self.lower_bound, self.upper_bound, axis = 1)
        
        elif self.action == 'flag':
            # create a column indicating the presence of outliers
            X_copy['outlier_flag'] = outliers.any(axis = 1)
        
        return X_copy
        
# ===============================
# Missing Values Handling
# ===============================

class MissingValuesHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values using strategies defined in config.py or manual overrides
    """
    def __init__(self, numerical_strategy =  None, categorical_strategy = None):
        """
        Parameters:
            - numerical_strategy (str): Strategy for numerical columns ('mean', 'median')
            - categorical_strategy (str): Strategy for numerical columns ('mode')
        """
        # use config default if no arguments are provided
        self.numerical_strategy =  numerical_strategy or cf.imputation_strategies['numerical']
        self.categorical_strategy =  categorical_strategy or cf.imputation_strategies['categorical']

    def fit(self, X, y = None):
        """
        Fits necessary statistics for imputation
        """
        self.numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns
        self.categorical_cols = X.select_dtypes(include = 'object').columns
        
        # fit numerical imputation
        if self.numerical_strategy == 'mean':
            self.num_fill_values = X[self.numerical_cols].mean()
        elif self.numerical_strategy == 'median':
            self.num_fill_values = X[self.numerical_cols].median()
        
        # fit categorical imputation
        if self.categorical_strategy == 'mode':
            self.cat_fill_values = X[self.categorical_cols].mode().iloc[0]
        
        return self
        
    def transform(self, X):
        """
        Applies imputation to missing values
        """
        
        X_copy = X.copy()
        
        # impute numerical columns
        X_copy[self.numerical_cols] = X_copy[self.numerical_cols].fillna(self.num_fill_values)
        
        # impute categorical columns
        X_copy[self.categorical_cols] = X_copy[self.categorical_cols].fillna(self.cat_fill_values)
        
        return X_copy
        
# ===============================
# Encoding Categorical
# ===============================

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Categorical feature encoding for binary, ordinal and nominal
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        """
        fitting if necessary
        """
        return self
    
    def transform(self, X):
        """
        Apply the enconding
        """
        X_copy = X.copy()
        
        # binary encoding
        for col, mapping in cf.binary_mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping)
                print(f'binary encoding applied to : {col}')
                
        # ordinal encoding
        for col, mapping in cf.ordinal_mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping)
                print(f'ordinal encoding applied to : {col}')
        
        # nominal encoding (one-hot)
        for col in cf.nominal_columns:
            if col in X_copy.columns:
                dummies = pd.get_dummies(X_copy[col], prefix = col)
                X_copy = pd.concat([X_copy, dummies], axis = 1)
                X_copy.drop(columns = [col], inplace = True)
                print(f'one - hot encoding applied to : {col}')
                
        return X_copy

# ===============================
# Scaling
# ===============================

class Scaling(BaseEstimator, TransformerMixin):
    """
    Scaling of set
    Scales features to range [0, 1] by default
    """
    
    def __init__(self, feature_range = (0, 1)):
        """
        Parameters:
            - feature_range (tuple): Desired range of tranformed data 
        """
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range = self.feature_range)
        self.numerical_cols = None
    
    def fit(self, X, y = None):
        """
        Fit the MinMaxScaler
        """
        self.numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns
        self.scaler.fit(X[self.numerical_cols])
        return self
        
    def transform(self, X):
        """
        Transform the numerical features using the Min-Max Scaling
        """
        
        X_copy = X.copy()
        
        # raise error if numerical columns not present
        required_columns = self.numerical_cols
        missing_columns = [col for col in required_columns if col not in X_copy.columns]
        
        if missing_columns:
            raise MissingColumnError(missing_columns)
        
        # apply scaling
        X_copy[self.numerical_cols] = self.scaler.transform(X_copy[self.numerical_cols])
        
        return X_copy
        
        
# ===============================
# Data Augmentation
# ===============================

class DataAugmentation(BaseEstimator, TransformerMixin):
    """
    Data Augmentation for tabular data
    """
    
    def __init__(self, method = 'smote', target_col = None, active = True, k_neighbors = 5, random_state = cf.random_state):
        """
        Parameters:
            - method (str): 'oversample' or 'smote'
            - target_col (str): target column name
            - active (bool): if False, does not apply Data Augmentation
            - k_neighbors (int): number of neighbors for SMOTE 
        """
        self.method = method
        self.target_col = target_col
        self.active = active
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampler = None
        
    def fit(self, X, y = None):
        if not self.active:
            return self
        
        if y is None:
            if self.target_col is None:
                raise ValueError('Must input and/or indicate target_col')
            y = X[self.target_col]
            X = X.drop(columns = [self.target_col])
        
        if self.method == 'smote':
            self.sampler = SMOTE(
                k_neighbors = self.k_neighbors,
                random_state = self.random_state
            )
        elif self.method == 'oversample':
            self.sampler = RandomOverSampler(random_state = self.random_state)
        else:
            raise ValueError('Method must be "smote" or "oversample"')

        self.sampler.fit(X, y)
        return self

    def transform(self, X, y = None):
        if not self.active:
            return (X, y) if y is not None else X
        
        if y is None:
            y = X[self.target_col]
            X = X.drop(columns = [self.target_col])
        
        X_res, y_res = self.sampler.fit_resample(X, y)
        
        # return both variables if downstream needs them
        if self.target_col is not None:
            X_res[self.target_col] = y_res
            return X_res
        return X_res, y_res # if target and features are inputed separately       



