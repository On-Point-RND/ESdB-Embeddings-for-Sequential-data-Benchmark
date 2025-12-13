"""Regression task implementation"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any
from omegaconf import DictConfig
import numpy as np

from ..core.base_classes import BaseTask
from ..core.types import TaskType

# Import CatBoost
from catboost import CatBoostRegressor

class RegressionTask(BaseTask):
    """Regression task implementation"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.optuna_config = config.get('optuna', {})
        self.use_optuna = self.optuna_config.get('enabled', False)
        
        # Define models including CatBoost
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'catboost': CatBoostRegressor(
                iterations=100,
                random_state=42, 
                verbose=False,
                thread_count=1
            )
        }
        
        # Search spaces for hyperparameter optimization
        self.search_spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'mlp': {
                'hidden_layer_sizes': [64, 128],
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
        """Create RandomizedSearchCV model with optimized thread control"""
        n_iter = self.optuna_config.get('n_trials', 10)
        cv = self.optuna_config.get('cv', 3)
        n_jobs = self.optuna_config.get('n_jobs', 4)
        max_threads = self.optuna_config.get('max_threads', 4)
        
        # Configure model-specific threading
        if model_name == 'catboost':
            # For CatBoost, use internal threading with limit
            catboost_threads = min(self.optuna_config.get('catboost_threads', max_threads), max_threads)
            if hasattr(base_model, 'set_params'):
                base_model.set_params(thread_count=catboost_threads)
            # Use n_jobs=1 for CatBoost to avoid process-level parallelism conflicts
            n_jobs = 1
            print(f"    CatBoost configured with {catboost_threads} threads")
        
        elif model_name == 'random_forest':
            # For Random Forest, limit n_jobs to avoid over-subscription
            n_jobs = min(n_jobs, max_threads)
        
        return RandomizedSearchCV(
            base_model,
            self.search_spaces[model_name],
            n_iter=n_iter,
            cv=cv,
            scoring='r2',  # Use R2 for regression
            random_state=42,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score='raise'
        )

    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute regression task"""
        if task_type != TaskType.REGRESSION:
            raise ValueError("This task only supports regression")

        print("Running regression task...")
        if self.use_optuna:
            n_jobs = self.optuna_config.get('n_jobs', 4)
            max_threads = self.optuna_config.get('max_threads', 4)
            print(f"Using hyperparameter optimization (n_jobs={n_jobs}, max_threads={max_threads})")
        
        print(f"Available models: {list(self.models.keys())}")
        
        results = {}

        for name, base_model in self.models.items():
            print(f"Training {name}...")
            
            # Use hyperparameter optimization if enabled
            if self.use_optuna:
                model = self._create_search_model(name, base_model)
            else:
                model = base_model
            
            # Train and evaluate
            model.fit(split_data['X_train'], split_data['y_train'])
            predictions = model.predict(split_data['X_test'])
            
            # Calculate metrics
            mse = mean_squared_error(split_data['y_test'], predictions)
            r2 = r2_score(split_data['y_test'], predictions)

            # Store results
            results[name] = {
                'mse': mse,
                'r2': r2,
                'predictions': predictions,
                'model': model
            }

            # Add CV results if available
            if self.use_optuna and hasattr(model, 'best_score_'):
                cv_r2 = model.best_score_
                results[name]['cv_r2'] = cv_r2
                results[name]['best_params'] = model.best_params_
                
                # Get CV standard deviation
                if hasattr(model, 'cv_results_') and 'std_test_score' in model.cv_results_:
                    cv_std = model.cv_results_['std_test_score'][model.best_index_]
                    results[name]['cv_r2_std'] = cv_std
                    
                    print(f"  {name}: CV R2 = {cv_r2:.4f} ± {cv_std:.4f}, Test R2 = {r2:.4f}, Test MSE = {mse:.4f}")
                else:
                    print(f"  {name}: CV R2 = {cv_r2:.4f}, Test R2 = {r2:.4f}, Test MSE = {mse:.4f}")
                    
            else:
                # For non-optimized models, we only have test scores
                results[name]['cv_r2'] = None
                results[name]['cv_r2_std'] = None
                print(f"  {name}: Test R2 = {r2:.4f}, Test MSE = {mse:.4f}")

        return results
