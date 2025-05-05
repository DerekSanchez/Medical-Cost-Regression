import src.config as cf
import src.utils as ut
import numpy as np
from importlib import import_module
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

def get_model(model_name):
    """
    Load model dynamically from config.py
    """
    model_name, class_name = cf.models[model_name].rsplit('.', 1)
    module = import_module(model_name)
    return getattr(module, class_name)()

def train_model(model_name, X_train, y_train, mode='manual', n_iter_random=50):
    """
    Trains a generic sklearn regression model.
    This function can tune hyperparameters using:
      - manual tuning
      - grid search
      - randomized search

    Parameters:
        - model_name (str): Model identifier to retrieve from get_model
        - X_train (pd.DataFrame): Training features
        - y_train (pd.Series): Continuous target variable
        - mode (str): 'manual', 'grid_search', or 'random_search'
        - n_iter_random (int): Number of iterations for RandomizedSearchCV

    Returns:
        dict: {
          'best_model': trained estimator,
          'best_params': optimal hyperparameters,
          'cv_train_score': {'mean': float, 'std': float},
          'cv_val_score': {'mean': float, 'std': float or None}
        }
    """
    # Log start
    ut.write_log(f"Start training (regression) of model: {model_name}")

    # Initialize model
    model = get_model(model_name)

    # Determine scoring metric key from config (e.g. 'mse', 'mae', 'r2')
    scoring_key = cf.scoring_methods[cf.scoring_mode]

    # Create a scorer for manual CV
    scorer = make_scorer(
        mean_squared_error if scoring_key == 'mse' else
        mean_absolute_error if scoring_key == 'mae' else
        r2_score,
        greater_is_better=(scoring_key == 'r2')
    )

    # Map to sklearn's CV scoring strings
    cv_scoring = (
        'neg_mean_squared_error' if scoring_key == 'mse' else
        'neg_mean_absolute_error' if scoring_key == 'mae' else
        'r2'
    )

    results = {}

    if mode == 'manual':
        # Manual parameter setting
        params = cf.manual_hyperparameters[model_name]
        model.set_params(**params)

        # Prepare CV
        kf = KFold(n_splits=cf.cv_folds, shuffle=True, random_state=cf.random_state)
        train_scores, val_scores = [], []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit and evaluate
            model.fit(X_tr, y_tr)
            train_scores.append(scorer(model, X_tr, y_tr))
            val_scores.append(scorer(model, X_val, y_val))

        # Aggregate
        results['cv_train_score'] = {'mean': np.mean(train_scores), 'std': np.std(train_scores)}
        results['cv_val_score']   = {'mean': np.mean(val_scores),   'std': np.std(val_scores)}

        # Final fit on all data
        model.fit(X_train, y_train)
        results['best_model']  = model
        results['best_params'] = params

    elif mode == 'grid_search':
        # Grid Search
        param_grid = cf.grid_param_distributions[model_name]
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cf.cv_folds,
            scoring=cv_scoring,
            n_jobs=-1,
            return_train_score=True
        )
        gs.fit(X_train, y_train)

        results['cv_train_score'] = {
            'mean': np.mean(gs.cv_results_['mean_train_score']),
            'std':  np.std(gs.cv_results_['mean_train_score'])
        }
        results['cv_val_score'] = {'mean': gs.best_score_, 'std': None}
        results['best_model']  = gs.best_estimator_
        results['best_params'] = gs.best_params_

    elif mode == 'random_search':
        # Randomized Search
        param_dist = cf.random_param_distributions[model_name]
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter_random,
            scoring=cv_scoring,
            cv=cf.cv_folds,
            random_state=cf.random_state,
            n_jobs=-1,
            return_train_score=True
        )
        rs.fit(X_train, y_train)

        results['cv_train_score'] = {
            'mean': np.mean(rs.cv_results_['mean_train_score']),
            'std':  np.std(rs.cv_results_['mean_train_score'])
        }
        results['cv_val_score'] = {'mean': rs.best_score_, 'std': None}
        results['best_model']  = rs.best_estimator_
        results['best_params'] = rs.best_params_

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'manual', 'grid_search', or 'random_search'.")

    # Log results
    ut.write_log(f"Best Hyperparameters: {results['best_params']}")
    ut.write_log(f"CV Train Score: {results['cv_train_score']}")
    ut.write_log(f"CV Validation Score: {results['cv_val_score']}")
    ut.write_log(f"Training complete for model: {model_name}")

    return results
