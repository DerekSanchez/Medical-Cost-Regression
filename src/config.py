
# ===============================
# Libraries and Configurations
# ===============================

# -------------------------------
# Libraries
# -------------------------------

from pathlib import Path
from scipy.stats import randint, uniform

# -------------------------------
# Paths
# -------------------------------

raw_data = 'raw_data.csv'
clean_data = 'clean_data.csv'
model_file = 'model.pkl'

current_path = Path(__file__).resolve().parent
root_path = current_path.parent

paths = {
    # folders
    'data' : root_path / 'data',
    'notebooks' : root_path / 'notebooks',
    'results' : root_path / 'results',
    'models' : root_path / 'models',
    'source' : root_path / 'src',
    # specific files
    'data_raw' : root_path / 'data' / 'raw_data.csv',
    'model' : root_path / 'models' / 'model.pkl',
    'logs' : root_path / 'results' / 'log.txt'   
}

print("project path:", root_path)
print("raw data path:", paths['data_raw'])
print("model path", paths['model'])
print("results path", paths['results'])


# ===============================
# Target Preprocessing
# ===============================

# No target preprocessing needed for this dataset

# ===============================
# Feature Preprocessing
# ===============================



# -------------------------------
# Feature Selection
# -------------------------------

num_features_to_drop = []

cat_features_to_drop = []

# -------------------------------
# Missing Values Handling
# -------------------------------

imputation_strategies = {
    "numerical" : "median",
    "categorical" : "mode"
}

# -------------------------------
# Encoding
# -------------------------------

binary_mappings = {
    'sex': {'male' : 1, 'female' : 0},
    'smoker': {'yes' : 1, 'no' : 0}
}

ordinal_mappings = {}

nominal_columns = [
    'region'
    ]



# -------------------------------
# Data Split
# -------------------------------

test_size = 0.2
random_state = 123


# ===============================
# Model Configuration
# ===============================

models = {
    'RandomForest' : 'sklearn.ensemble.RandomForestRegressor',
    'GradientBoosting' : 'sklearn.ensemble.GradientBoostingRegressor',
    'LinearRegression' : 'sklearn.linear_model.LinearRegression',
    'SVR' : 'sklearn.svm.SVR'
}


# ===============================
# Hyperparameters
# ===============================

# -------------------------------
# Model Hyperparameters
# -------------------------------

random_param_distributions = {
    'RandomForest': {
        'n_estimators': randint(50, 500),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20)
    },
    'GradientBoosting': {
        'learning_rate': uniform(0.01, 0.2),
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 15),
        'subsample': uniform(0.5, 0.5)
    },
    'LinearRegression': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'SVR': {
        'C': uniform(0.01, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': uniform(0.001, 1)
    }
}

grid_param_distributions = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.5, 0.75, 1.0]
    },
    'LinearRegression': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'SVR': {
        'C': [9.3, 9.6, 9.9, 10, 10.1],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.99, 1.0, 1.1]
    }
}

manual_hyperparameters = {
    'RandomForest': {
        'n_estimators': 364,
        'max_depth': 5,
        'min_samples_split': 4,
        'min_samples_leaf': 12
    },
    'GradientBoosting': {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'max_depth': 5,
        'subsample': 0.75
    },
    'LinearRegression': {
        'fit_intercept': True
    },
    'SVR': {
        'C': 10.1,
        'kernel': 'poly',
        'gamma': 1.1,
        'degree': 5
    }
}

# -------------------------------
# Enviroment Hyperparameters
# -------------------------------

# cross-validation configuration
cv_folds = 10

# optimization scoring
scoring_methods = {
    'default': 'r2',
    'alternative': 'neg_mean_squared_error'
}

scoring_mode = 'default'


# ===============================
# Other Configurations
# ===============================

# -------------------------------
# Plot Configuration
# -------------------------------

plot_style = 'seaborn-darkgrid'
fig_size = (10, 6)