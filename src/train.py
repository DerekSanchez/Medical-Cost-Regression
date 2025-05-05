import src.config as cf
import src.utils as ut
import numpy as np
from importlib import import_module
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score


def get_model(model_name):
    """
    Load model dynamically from config.py
    """
    model_name, class_name = cf.models[model_name].rsplit('.', 1)
    module = import_module(model_name)
    return getattr(module, class_name)()

def train_model(model_name, X_train, y_train, mode='manual', n_inter_random=50):
    """
    Trains a generic sklearn model.
    This function has the flexibility to tune the hyperparameters using
        - manual tuning 
        - grid search 
        - randomized search
        
    Parameters:
        - model_name (str): Model name to train
        - X_train (pd.DataFrame): Test set
        - y_train (pd.Series): Target variable
        - mode (str): 'manual, 'grid_search' or 'random_search'
        - n_inter_random (int): Number of combinations to try out in Random Search
        
    Returns:
        dict: Dictionary with model trained, using best parameters and metrics of CV
    """
    
    # document a log of the model training
    ut.write_log(f'Start training of model: {model_name}')
    
    # settings
    model = get_model(model_name)
    scoring = cf.scoring_methods[cf.scoring_mode]  # define scoring metric
    scorer = make_scorer(
        mean_squared_error if scoring == 'mse' else
        mean_absolute_error if scoring == 'mae' else
        r2_score, greater_is_better=(scoring != 'mse' and scoring != 'mae')
    )  # allows personalization of metric through variable score instance
    results = {}
    
    if mode == 'manual':
        # manual hyperparameter tuning
        params = cf.manual_hyperparameters[model_name]
        
        model.set_params(**params)
        
        # personalized cross validation
        train_scores = []
        val_scores = []
        skf = StratifiedKFold(n_splits=cf.cv_folds, shuffle=True, random_state=cf.random_state)
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            # divide X and y into train and val, respectively
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # fit the model on train cv
            model.fit(X_train_cv, y_train_cv)
            
            # evaluate model on both train and validation
            train_scores_temp = scorer(model, X_train_cv, y_train_cv)
            val_scores_temp = scorer(model, X_val_cv, y_val_cv)
            
            # append results on separate lists
            train_scores.append(train_scores_temp)
            val_scores.append(val_scores_temp)
        
        # calculate statistics for evaluation results
        results['cv_train_score'] = {
            'mean': np.mean(train_scores),
            'std': np.std(train_scores)
        }
         
        results['cv_val_score'] = {
            'mean': np.mean(val_scores),
            'std': np.std(val_scores)
        }   
        
        # fit model on train set
        model.fit(X_train, y_train)
        results['best_model'] = model
        results['best_params'] = params
        
    elif mode == 'grid_search':
        # define the grid space
        param_grid = cf.grid_param_distributions[model_name]
        
        # use grid search, evaluate with CV and fit the model
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cf.cv_folds, 
            scoring=scoring, 
            n_jobs=-1,
            return_train_score=True)
        grid_search.fit(X_train, y_train)
        
        # store train results
        results['cv_train_score'] = {
            'mean': np.mean(grid_search.cv_results_['mean_train_score']),
            'std': np.std(grid_search.cv_results_['mean_train_score'])
        }
        
        # store validation results
        results['cv_val_score'] = {
            'mean': grid_search.best_score_,
            'std': None
        }
        
        # get best results
        results['best_model'] = grid_search.best_estimator_
        results['best_params'] = grid_search.best_params_
        
    elif mode == 'random_search':
        # define the random range
        param_dist = cf.random_param_distributions[model_name]
        
        # use randomized search, evaluate with CV and fit the model
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_inter_random,
            scoring=scoring,
            cv=cf.cv_folds,
            random_state=cf.random_state,
            n_jobs=-1,
            return_train_score=True
        )
        random_search.fit(X_train, y_train)
        
        # store train results
        results['cv_train_score'] = {
            'mean': np.mean(random_search.cv_results_['mean_train_score']),
            'std': np.std(random_search.cv_results_['mean_train_score'])
        }
        
        # store validation results
        results['cv_val_score'] = {
            'mean': random_search.best_score_,
            'std': None
        }
        
        # get best results
        results['best_model'] = random_search.best_estimator_
        results['best_params'] = random_search.best_params_
    
    # document training results log
    ut.write_log(f"Best Hyperparameters: {results['best_params']}")
    ut.write_log(f"CV Train Score: {results['cv_train_score']}")
    ut.write_log(f"CV Validation Score: {results['cv_val_score']}")
    ut.write_log(f"Training complete for model: {model_name}")
    
    return results
