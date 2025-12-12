"""Classification task implementation"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Dict, Any
from omegaconf import DictConfig
import numpy as np

import optuna
import optuna.logging
from optuna.trial import Trial

from ..core.base_classes import BaseTask
from ..core.types import TaskType
from catboost import CatBoostClassifier

class ClassificationTask(BaseTask):
    """Classification task implementation"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.optuna_config = config.get('optuna', {})
        self.use_optuna = self.optuna_config.get('enabled', False)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        # Define models including CatBoost
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'mlp': MLPClassifier(random_state=42, max_iter=1000),
            'catboost': CatBoostClassifier(random_state=42, verbose=False, thread_count=1)
        }
        
        # Search spaces for hyperparameter optimization (kept for reference)
        self.search_spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'mlp': {
                'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64)],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1],
            },
            'catboost': {
                'iterations': [50, 100, 150, 200],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
            }
        }

    def _create_search_model(self, model_name, base_model):
        """Create Optuna study instead of RandomizedSearchCV"""
        if not self.use_optuna:
            # If Optuna is disabled, return the base model with default params
            if model_name == 'random_forest':
                base_model.set_params(n_estimators=100)
            elif model_name == 'catboost':
                base_model.set_params(iterations=100)
            return base_model
        
        n_trials = self.optuna_config.get('n_trials', 10)
        cv_folds = self.optuna_config.get('cv', 3)
        n_jobs = self.optuna_config.get('n_jobs', 4)
        
        print(f"    Using Optuna with {n_trials} trials, {cv_folds}-fold CV")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Store the study and config to use later during fit
        return {
            'study': study,
            'model_name': model_name,
            'base_model': base_model,
            'cv_folds': cv_folds,
            'n_trials': n_trials,
            'n_jobs': n_jobs
        }

    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute classification task"""
        if task_type != TaskType.CLASSIFICATION:
            raise ValueError("This task only supports classification")

        print("Running classification task...")
        
        results = {}

        for name, base_model in self.models.items():
            print(f"Training {name}...")
            
            # Get model (either base model or Optuna study config)
            model = self._create_search_model(name, base_model)
            
            if self.use_optuna:
                # Optuna optimization
                study = model['study']
                model_name = model['model_name']
                cv_folds = model['cv_folds']
                n_trials = model['n_trials']
                
                # Define objective function for Optuna
                def objective(trial):
                    # Get hyperparameters based on model type
                    if model_name == 'random_forest':
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'max_depth': trial.suggest_int('max_depth', 3, 20),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                            'n_jobs': 1  # Run each trial single-threaded
                        }
                    elif model_name == 'mlp':
                        params = {
                            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64,), (128,), (64, 64), (128, 64)]),
                            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
                        }
                    elif model_name == 'catboost':
                        params = {
                            'iterations': trial.suggest_int('iterations', 50, 200),
                            'depth': trial.suggest_int('depth', 4, 10),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 9),
                            'thread_count': 1
                        }
                    
                    # Create model with suggested parameters
                    if model_name == 'random_forest':
                        current_model = RandomForestClassifier(random_state=42, **params)
                    elif model_name == 'mlp':
                        current_model = MLPClassifier(random_state=42, max_iter=1000, **params)
                    elif model_name == 'catboost':
                        current_model = CatBoostClassifier(random_state=42, verbose=False, **params)
                    
                    # Cross-validation with StratifiedKFold
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    scores = cross_val_score(
                        current_model, 
                        split_data['X_train'], 
                        split_data['y_train'], 
                        cv=cv, 
                        scoring='accuracy',
                        n_jobs=1  # Single thread per CV fold
                    )
                    
                    # Calculate mean and std
                    cv_mean = np.mean(scores)
                    cv_std = np.std(scores)
                    
                    # Store std as user attribute for analysis
                    trial.set_user_attr("cv_std", cv_std)
                    
                    return cv_mean
                
                # Run Optuna optimization
                study.optimize(objective, n_trials=n_trials, n_jobs=model['n_jobs'])
                
                # Get best parameters and CV results
                best_params = study.best_params
                best_value = study.best_value
                
                # Get the best trial to extract CV std
                best_trial = study.best_trial
                best_cv_std = best_trial.user_attrs.get("cv_std", 0.0)
                
                # Create final model with best parameters
                if name == 'random_forest':
                    final_model = RandomForestClassifier(random_state=42, **best_params)
                elif name == 'mlp':
                    final_model = MLPClassifier(random_state=42, max_iter=1000, **best_params)
                elif name == 'catboost':
                    final_model = CatBoostClassifier(random_state=42, verbose=False, **best_params)
                
                # Train on full training data
                final_model.fit(split_data['X_train'], split_data['y_train'])
                predictions = final_model.predict(split_data['X_test'])
                test_accuracy = accuracy_score(split_data['y_test'], predictions)
                
                # Store results
                results[name] = {
                    'accuracy': test_accuracy,  # Test accuracy
                    'predictions': predictions,
                    'model': final_model,
                    'cv_accuracy': best_value,  # CV mean accuracy
                    'cv_accuracy_std': best_cv_std,  # CV std
                    'best_params': best_params,
                    'study': study  # Keep study for analysis if needed
                }
                
                print(f"  {name}: CV = {best_value:.4f} ± {best_cv_std:.4f}, Test = {test_accuracy:.4f}")
                
            else:
                # Original non-Optuna code
                model.fit(split_data['X_train'], split_data['y_train'])
                predictions = model.predict(split_data['X_test'])
                test_accuracy = accuracy_score(split_data['y_test'], predictions)
                
                results[name] = {
                    'accuracy': test_accuracy,
                    'predictions': predictions,
                    'model': model,
                    'cv_accuracy': None,
                    'cv_accuracy_std': None
                }
                
                print(f"  {name}: Test = {test_accuracy:.4f}")

        return results
