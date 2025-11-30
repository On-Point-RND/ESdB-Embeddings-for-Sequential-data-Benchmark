"""Anomaly detection task implementation"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Any
from omegaconf import DictConfig
import numpy as np

from ..core.base_classes import BaseTask
from ..core.types import TaskType

# Import CatBoost
from catboost import CatBoostClassifier

class AnomalyDetectionTask(BaseTask):
    """Anomaly detection task implementation"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.optuna_config = config.get('optuna', {})
        self.use_optuna = self.optuna_config.get('enabled', False)
        
        # Define models including CatBoost with class weights
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                random_state=42, 
                max_iter=1000
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                random_state=42, 
                verbose=False,
                thread_count=1,
                auto_class_weights='Balanced'
            )
        }
        
        # Search spaces for hyperparameter optimization with F1 scoring
        self.search_spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
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
                'auto_class_weights': ['Balanced', 'SqrtBalanced']  # Remove None, use valid options
            }
        }

    def _create_search_model(self, model_name, base_model):
        """Create RandomizedSearchCV model with F1 scoring and optimized thread control"""
        n_iter = self.optuna_config.get('n_trials', 10)
        cv = self.optuna_config.get('cv', 3)
        n_jobs = self.optuna_config.get('n_jobs', 4)
        max_threads = self.optuna_config.get('max_threads', 4)
        
        # Configure model-specific threading
        if model_name == 'catboost':
            catboost_threads = min(self.optuna_config.get('catboost_threads', max_threads), max_threads)
            if hasattr(base_model, 'set_params'):
                base_model.set_params(thread_count=catboost_threads)
            n_jobs = 1
            print(f"    CatBoost configured with {catboost_threads} threads")
        
        elif model_name == 'random_forest':
            n_jobs = min(n_jobs, max_threads)
        
        return RandomizedSearchCV(
            base_model,
            self.search_spaces[model_name],
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            random_state=42,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score='raise'
        )

    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute anomaly detection task"""
        if task_type != TaskType.ANOMALY_DETECTION:
            raise ValueError("This task only supports anomaly detection")

        print("Running anomaly detection task...")
        if self.use_optuna:
            n_jobs = self.optuna_config.get('n_jobs', 4)
            max_threads = self.optuna_config.get('max_threads', 4)
            print(f"Using hyperparameter optimization with F1 scoring (n_jobs={n_jobs}, max_threads={max_threads})")
        
        print(f"Available models: {list(self.models.keys())}")
        
        # Print class distribution for reference
        train_anomaly_ratio = split_data['y_train'].mean()
        test_anomaly_ratio = split_data['y_test'].mean()
        print(f"Class distribution - Train: {train_anomaly_ratio:.2%} anomalies, Test: {test_anomaly_ratio:.2%} anomalies")
        
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
            
            # Calculate anomaly detection metrics
            f1 = f1_score(split_data['y_test'], predictions)
            precision = precision_score(split_data['y_test'], predictions)
            recall = recall_score(split_data['y_test'], predictions)
            
            # Calculate AUC if possible
            auc = 0.0
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(split_data['X_test'])[:, 1]
                auc = roc_auc_score(split_data['y_test'], y_scores)

            # Store results
            results[name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'predictions': predictions,
                'model': model
            }

            # Add CV results if available
            if self.use_optuna and hasattr(model, 'best_score_'):
                cv_f1 = model.best_score_
                results[name]['cv_f1'] = cv_f1
                results[name]['best_params'] = model.best_params_
                
                # Get CV standard deviation
                if hasattr(model, 'cv_results_') and 'std_test_score' in model.cv_results_:
                    cv_std = model.cv_results_['std_test_score'][model.best_index_]
                    results[name]['cv_f1_std'] = cv_std
                    
                    print(f"  {name}: CV F1 = {cv_f1:.4f} ± {cv_std:.4f}, Test F1 = {f1:.4f}, AUC = {auc:.4f}")
                else:
                    print(f"  {name}: CV F1 = {cv_f1:.4f}, Test F1 = {f1:.4f}, AUC = {auc:.4f}")
                    
            else:
                results[name]['cv_f1'] = None
                results[name]['cv_f1_std'] = None
                print(f"  {name}: Test F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, AUC = {auc:.4f}")

        return results
