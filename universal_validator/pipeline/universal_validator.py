"""Main pipeline orchestrator with task routing"""
import os
from typing import Dict, Any
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from ..splitters.standard_splitter import StandardSplitter
from ..tasks.classification_task import ClassificationTask
from ..tasks.regression_task import RegressionTask
from ..tasks.forecast_task import ForecastTask
from ..tasks.anomaly_detection_task import AnomalyDetectionTask
from ..types import TaskType

class UniversalValidator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.splitter_registry = {'standard': StandardSplitter(self.config.splitting)}
        self.task_registry = {
            TaskType.CLASSIFICATION: ClassificationTask(self.config),
            TaskType.REGRESSION: RegressionTask(self.config),
            TaskType.ANOMALY_DETECTION: AnomalyDetectionTask(self.config),
            TaskType.FORECAST: ForecastTask(self.config)
        }
    
    def run_pipeline(self, dataset_name: str = 'age', task_type: TaskType = TaskType.CLASSIFICATION,
                    splitter_name: str = None, embeddings_path: str = None) -> Dict[str, Any]:
        if embeddings_path is None:
            embeddings_path = f'embeddings_{dataset_name}_coles.parquet'
        print(f"embeddings_path: {embeddings_path}")
        print(f"Starting {dataset_name} pipeline, Task: {task_type.value}")
        
        if not os.path.exists(embeddings_path):
            print(f"No embeddings at {embeddings_path}")
            return {}
        
        print(f"Loading embeddings from {embeddings_path}")
        parquet_df = pd.read_parquet(embeddings_path).dropna()
        embeddings_df = pd.DataFrame(np.stack(list(parquet_df['embedding']), axis=0))
        embeddings_df.columns = [f"embed_{i}" for i in range(embeddings_df.shape[1])]
        embeddings_df = pd.concat([parquet_df.drop('embedding', axis=1).reset_index(drop=True), embeddings_df], axis=1)
        
        targets_df = self._get_targets(dataset_name, task_type, embeddings_df)
        splitter = self.splitter_registry[splitter_name]
        split_data = splitter.split(embeddings_df, targets_df, task_type)
        task = self.task_registry[task_type]
        results = task.execute(split_data)
        report = self._generate_report(dataset_name, splitter_name, task_type, results)
        print("Pipeline completed successfully!")
        return report
    
    def _get_targets(self, dataset_name: str, task_type: TaskType, embeddings_df: pd.DataFrame):
        if task_type == TaskType.CLASSIFICATION:
            return embeddings_df['post_target'].copy()
        elif task_type == TaskType.FORECAST:
            return embeddings_df['post_forecast_target'].copy()
        elif task_type == TaskType.ANOMALY_DETECTION:
            return embeddings_df['post_anomaly_target'].copy()
        elif task_type == TaskType.REGRESSION:
            amount_cols = [col for col in embeddings_df.columns if 'post_amount' in col]
            if not amount_cols:
                raise ValueError("No amount column for regression")
            amount_col = sorted(amount_cols)[0]
            target_values = embeddings_df[amount_col].values
            
            if hasattr(target_values[0], '__len__'):
                if 'age' in dataset_name:
                    return pd.DataFrame([np.log(np.median(v)) for v in target_values], columns=['target'])
                elif 'zvuk' in dataset_name:
                    return pd.DataFrame([np.nanmedian(v) for v in target_values], columns=['target'])
                else:
                    return pd.DataFrame([np.log1p(np.nanmedian(v+1e-10)) for v in target_values], columns=['target'])
            else:
                return pd.DataFrame(embeddings_df[amount_col]).rename(columns={amount_col: 'target'})
        else:
            raise NotImplementedError(f"Task type {task_type}")
    
    def _generate_report(self, dataset_name, splitter_name, task_type, results):
        if not results:
            return {'dataset': dataset_name, 'splitter': splitter_name, 
                    'task_type': task_type.value, 'error': 'No models trained'}

        metric_map = {
            TaskType.CLASSIFICATION: 'accuracy',
            TaskType.REGRESSION: 'r2',
            TaskType.ANOMALY_DETECTION: 'auc',
            TaskType.FORECAST: 'r2'  # r2 for forecast
        }

        metric = metric_map[task_type]
        best_model = max(results.keys(), key=lambda x: results[x][metric])

        return {
            'dataset': dataset_name, 'splitter': splitter_name, 'task_type': task_type.value,
            'best_model': best_model, f'best_{metric}': results[best_model][metric],
            'all_results': results, 'timestamp': pd.Timestamp.now().isoformat()
        }