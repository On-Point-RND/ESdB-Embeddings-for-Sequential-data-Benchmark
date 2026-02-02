"""Base task implementation"""
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig, OmegaConf
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import optuna
import optuna.logging
from sklearn.base import BaseEstimator

from ..types import TaskType

class BaseTask:
    """Base class for all downstream tasks with common functionality"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Получаем downstream config
        self.downstream_config = self.config.get('downstream', OmegaConf.create({}))
        
        # Получаем optuna config
        self.optuna_config = self.downstream_config.get('optuna', OmegaConf.create({}))
        self.use_optuna = self.optuna_config.get('enabled', False)
        
        # Получаем имя задачи
        self.task_name = self._get_task_name()
        
        # Получаем конфигурацию для задачи
        self.task_config = self._get_task_specific_config()
        
        # Получаем конфигурацию параметров для Optuna
        self.optuna_params_config = self._get_optuna_params_config()
        
        # Инициализируем модели
        self.models = self._init_models()
        
        # Если модели не загрузились, используем дефолтные
        if not self.models:
            self.models = self._get_default_models()
            if self.models:
                print(f"Info: Using default models for {self.task_name}")
        
        # Устанавливаем уровень логирования Optuna
        if self.use_optuna:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    def _get_task_specific_config(self) -> DictConfig:
        """Get task-specific configuration"""
        # Сначала пробуем получить из downstream.optuna.<task_name>
        if OmegaConf.is_dict(self.downstream_config) and 'optuna' in self.downstream_config:
            task_config = self.downstream_config.optuna.get(self.task_name, OmegaConf.create({}))
            if task_config and len(task_config) > 0:
                return task_config
        
        # Пробуем получить напрямую из downstream.<task_name>
        task_config = self.downstream_config.get(self.task_name, OmegaConf.create({}))
        if task_config and len(task_config) > 0:
            return task_config
        
        # Если ничего не нашли, возвращаем пустой конфиг
        return OmegaConf.create({})
    
    def _get_optuna_params_config(self) -> DictConfig:
        """Get Optuna parameters configuration"""
        # Сначала пробуем получить из downstream.optuna.<task_name>.optuna_params
        if OmegaConf.is_dict(self.task_config) and 'optuna_params' in self.task_config:
            return self.task_config.optuna_params
        
        # Пробуем получить из отдельного раздела
        optuna_params_config = self.downstream_config.get('optuna_params', OmegaConf.create({}))
        
        # Если ничего не нашли, возвращаем пустой конфиг
        return optuna_params_config
    
    def _get_task_name(self) -> str:
        """Get task name for config lookup - MUST be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_task_name")
    
    def _init_models(self) -> Dict[str, BaseEstimator]:
        """Initialize models from configuration"""
        models_config = self.task_config.get('models', OmegaConf.create({}))
        
        # Если models_config пустой, возвращаем пустой словарь
        if not models_config or (OmegaConf.is_dict(models_config) and len(models_config) == 0):
            return {}
        
        models = {}
        
        # Проходим по всем моделям в конфиге
        for model_name, model_params in models_config.items():
            try:
                # Получаем класс модели
                model_class = self._import_model_class(model_params['class'])
                
                # Получаем параметры модели
                params_dict = {}
                if 'params' in model_params:
                    # Конвертируем OmegaConf в обычный dict
                    params_dict = dict(model_params.params)
                
                # Создаем модель
                models[model_name] = model_class(**params_dict)
                
            except Exception as e:
                print(f"Warning: Failed to initialize model {model_name} for {self.task_name}: {e}")
                continue
        
        return models
    
    def _import_model_class(self, class_path: str):
        """Dynamically import model class from string"""
        module_name, class_name = class_path.rsplit('.', 1)
        
        # Динамически импортируем модуль
        if 'sklearn' in module_name:
            if 'MLPClassifier' in class_name:
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier
            elif 'MLPRegressor' in class_name:
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor
        elif 'catboost' in module_name:
            if 'CatBoostClassifier' in class_name:
                from catboost import CatBoostClassifier
                return CatBoostClassifier
            elif 'CatBoostRegressor' in class_name:
                from catboost import CatBoostRegressor
                return CatBoostRegressor
        
        raise ImportError(f"Cannot import {class_path}")
    
    def _get_default_models(self) -> Dict[str, BaseEstimator]:
        """Get default models if none are configured - should be overridden by subclasses"""
        return {}
    
    def _configure_model_threading(self, model: BaseEstimator, model_name: str) -> BaseEstimator:
        """Configure threading for different model types"""
        max_threads = self.optuna_config.get('max_threads', 4)
        
        if model_name == 'catboost':
            catboost_threads = min(self.optuna_config.get('catboost_threads', max_threads), max_threads)
            if hasattr(model, 'set_params'):
                model.set_params(thread_count=catboost_threads)
            print(f"    CatBoost configured with {catboost_threads} threads")
        
        return model
    
    def _create_optuna_study(self, direction: str = "maximize") -> optuna.Study:
        """Create Optuna study for hyperparameter optimization"""
        # Получаем настройки сэмплера
        sampler_type = self.optuna_config.get('sampler', 'tpe')
        enable_pruning = self.optuna_config.get('pruning', True)
        
        # Выбор сэмплера
        if sampler_type == 'random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler_type == 'grid':
            sampler = optuna.samplers.GridSampler()
        else:  # tpe по умолчанию
            n_startup_trials = self.optuna_config.get('n_startup_trials', 10)
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=n_startup_trials)
        
        # Выбор пранера
        if enable_pruning:
            n_startup = self.optuna_config.get('pruning_startup', 5)
            n_warmup = self.optuna_config.get('pruning_warmup', 10)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup)
        else:
            pruner = optuna.pruners.NopPruner()
        
        return optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
    
    def _get_default_scoring(self) -> str:
        """Get default scoring metric"""
        return 'accuracy'
    
    def _get_optimization_direction(self) -> str:
        """Get optimization direction (maximize/minimize)"""
        return "maximize"
    
    def _get_cv_splitter(self, cv_folds: int):
        """Get appropriate CV splitter based on task type"""
        task_type = self._get_supported_task_type()
        
        if task_type in [TaskType.CLASSIFICATION, TaskType.ANOMALY_DETECTION]:
            # Для классификации используем StratifiedKFold
            return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            # Для регрессии и прогнозирования используем обычный KFold
            return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def _calculate_metrics(self, model: BaseEstimator, X_test: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, float]:
        """Calculate task-specific metrics - MUST be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _calculate_metrics")
    
    def _print_task_info(self, split_data: Dict[str, Any]) -> None:
        """Print task-specific information (optional override)"""
        # Default implementation - can be overridden
        pass
    
    def _get_optuna_params(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        """Get Optuna parameters for a model from configuration"""
        # Базовые параметры, которые всегда присутствуют
        base_params = {
            'random_state': 42,
            'verbose': False
        }
        
        # Получаем конфигурацию параметров для этой модели
        model_params_config = self.optuna_params_config.get(model_name, OmegaConf.create({}))
        
        if not model_params_config or len(model_params_config) == 0:
            # Если нет конфигурации, возвращаем базовые параметры
            return base_params
        
        params = {}
        
        for param_name, param_config in model_params_config.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                # Целочисленный параметр
                low = param_config.get('low', 0)
                high = param_config.get('high', 100)
                step = param_config.get('step', 1)
                log = param_config.get('log', False)
                
                if step == 1 and not log:
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    # Для нестандартных шагов используем categorical
                    values = list(range(low, high + 1, step))
                    params[param_name] = trial.suggest_categorical(param_name, values)
                    
            elif param_type == 'float':
                # Вещественный параметр
                low = param_config.get('low', 0.0)
                high = param_config.get('high', 1.0)
                log = param_config.get('log', False)
                
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                
            elif param_type == 'categorical':
                # Категориальный параметр
                choices = param_config.get('choices', [])
                if isinstance(choices, list) and len(choices) > 0:
                    params[param_name] = trial.suggest_categorical(param_name, choices)
                    
            elif param_type == 'uniform':
                # Равномерное распределение (для обратной совместимости)
                low = param_config.get('low', 0.0)
                high = param_config.get('high', 1.0)
                params[param_name] = trial.suggest_uniform(param_name, low, high)
                
            elif param_type == 'loguniform':
                # Лог-равномерное распределение
                low = param_config.get('low', 0.0001)
                high = param_config.get('high', 0.1)
                params[param_name] = trial.suggest_loguniform(param_name, low, high)
        
        # Объединяем с базовыми параметрами
        all_params = {**base_params, **params}
        
        # Добавляем специфичные параметры для моделей
        if model_name == 'catboost':
            all_params['thread_count'] = 1
            # Убираем verbose, так как уже есть в base_params
            if 'verbose' in all_params:
                all_params['verbose'] = False
        
        return all_params
    
    def _train_with_optuna(self, model_name: str, split_data: Dict[str, Any]) -> tuple:
        """Train model with Optuna optimization"""
        cv_folds = self.optuna_config.get('cv', 3)
        n_trials = self.optuna_config.get('n_trials', 10)
        n_jobs = self.optuna_config.get('n_jobs', 4)
        
        # Configure threading for the base model
        base_model = self.models[model_name]
        base_model = self._configure_model_threading(base_model, model_name)
        
        study = self._create_optuna_study(self._get_optimization_direction())
        
        def objective(trial):
            # Get parameters from configuration
            params = self._get_optuna_params(model_name, trial)
            
            # Create model with suggested parameters
            model_class = type(base_model)
            current_model = model_class(**params)
            
            # Configure threading for this trial
            current_model = self._configure_model_threading(current_model, model_name)
            
            # Get appropriate CV splitter
            cv = self._get_cv_splitter(cv_folds)
            
            # Cross-validation
            scores = self._cross_validate_model(current_model, split_data, cv)
            
            cv_mean = np.mean(scores)
            cv_std = np.std(scores)
            trial.set_user_attr("cv_std", cv_std)
            
            return cv_mean
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Create final model with best parameters
        final_model = type(base_model)(**best_params)
        final_model = self._configure_model_threading(final_model, model_name)
        final_model.fit(split_data['X_train'], split_data['y_train'])
        
        return final_model, {
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }
    
    def _cross_validate_model(self, model: BaseEstimator, split_data: Dict[str, Any], 
                            cv) -> List[float]:
        """Perform cross-validation"""
        from sklearn.model_selection import cross_val_score
        return cross_val_score(
            model,
            split_data['X_train'],
            split_data['y_train'],
            cv=cv,
            scoring=self._get_default_scoring(),
            n_jobs=1  # Single thread per CV fold
        )
    
    def _train_simple(self, model_name: str, split_data: Dict[str, Any]) -> tuple:
        """Train model without hyperparameter optimization"""
        model = self.models[model_name]
        model = self._configure_model_threading(model, model_name)
        model.fit(split_data['X_train'], split_data['y_train'])
        return model, None
    
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]) -> BaseEstimator:
        """Create model instance with given parameters"""
        model_class = type(self.models[model_name])
        model = model_class(**params)
        return self._configure_model_threading(model, model_name)
    
    def _get_supported_task_type(self) -> TaskType:
        """Get supported task type - MUST be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_supported_task_type")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models for this task"""
        return list(self.models.keys())
    
    def execute(self, split_data: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Execute task with common workflow"""
        # Validate task type
        if task_type != self._get_supported_task_type():
            raise ValueError(f"This task only supports {self._get_supported_task_type()}")
        
        print(f"Running {self._get_supported_task_type().value} task...")
        
        # Print task-specific info
        self._print_task_info(split_data)
        
        # Check if optimization is enabled
        if self.use_optuna:
            print(f"Using Optuna hyperparameter optimization")
            print(f"  n_trials: {self.optuna_config.get('n_trials', 10)}")
            print(f"  cv: {self.optuna_config.get('cv', 3)}")
            print(f"  n_jobs: {self.optuna_config.get('n_jobs', 4)}")
            print(f"  max_threads: {self.optuna_config.get('max_threads', 4)}")
            print(f"  sampler: {self.optuna_config.get('sampler', 'tpe')}")
        else:
            print(f"Using models with default parameters")
        
        available_models = self._get_available_models()
        if not available_models:
            print("Warning: No models available for this task!")
            return {}
        
        print(f"Available models: {available_models}")
        
        results = {}
        
        for name in available_models:
            print(f"Training {name}...")
            
            # Train model
            if self.use_optuna:
                model, cv_results = self._train_with_optuna(name, split_data)
            else:
                model, cv_results = self._train_simple(name, split_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(model, split_data['X_test'], split_data['y_test'])
            
            # Store results
            results[name] = {
                **metrics,
                'predictions': model.predict(split_data['X_test']),
                'model': model,
                'cv_results': cv_results
            }
            
            # Print results
            self._print_model_results(name, metrics, cv_results)
        
        return results
    
    def _print_model_results(self, model_name: str, metrics: Dict[str, float], 
                           cv_results: Optional[Dict[str, Any]]) -> None:
        """Print model results"""
        if cv_results and 'best_score' in cv_results:
            cv_score = cv_results.get('best_score', 0)
            cv_std = 0
            if 'study' in cv_results and cv_results['study'].best_trial:
                cv_std = cv_results['study'].best_trial.user_attrs.get("cv_std", 0)
            
            if cv_std:
                print(f"  {model_name}: CV = {cv_score:.4f} ± {cv_std:.4f}", end="")
            else:
                print(f"  {model_name}: CV = {cv_score:.4f}", end="")
        else:
            print(f"  {model_name}:", end="")
        
        # Print metrics
        for metric_name, value in metrics.items():
            print(f", {metric_name} = {value:.4f}", end="")
        print()
