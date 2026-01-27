"""Main pipeline orchestrator with task routing"""
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..core.types import TaskType
from ..core.base_classes import BaseDataset, BaseEmbedder, BaseSplitter, BaseTask
from ..splitters.standard_splitter import StandardSplitter
from ..tasks.classification_task import ClassificationTask
from ..tasks.regression_task import RegressionTask
from ..tasks.forecast_task import ForecastTask
from ..tasks.anomaly_detection_task import AnomalyDetectionTask
from scipy.stats import boxcox


class UniversalValidator:
    """Main class that orchestrates the entire validation pipeline with task routing"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.splitter_registry = self._initialize_splitters()
        self.task_registry = self._initialize_tasks()

    def _initialize_splitters(self) -> Dict[str, BaseSplitter]:
        """Initialize all available data splitters"""
        return {
            'standard': StandardSplitter(self.config.splitting),
        }

    def _initialize_tasks(self) -> Dict[TaskType, BaseTask]:
        """Initialize all available tasks"""
        return {
            TaskType.CLASSIFICATION: ClassificationTask(self.config.downstream),
            TaskType.REGRESSION: RegressionTask(self.config.downstream),
            TaskType.ANOMALY_DETECTION: AnomalyDetectionTask(self.config.downstream),
            TaskType.FORECAST: ForecastTask(self.config.downstream)
        }

    def run_pipeline(self,
                    dataset_name: str = 'age',
                    task_type: TaskType = TaskType.CLASSIFICATION,
                    splitter_name: str = None,
                    embeddings_path: str = None) -> Dict[str, Any]:
        """Run complete validation pipeline with task routing"""
        
        if embeddings_path is None:
            embeddings_path = f'embeddings_{dataset_name}_coles.parquet'
        print('embeddings_path', embeddings_path)
        print(f"Starting {dataset_name} pipeline")
        print(f"  Task: {task_type.value}")
        print(f"  Splitter: {splitter_name}")
        print("=" * 60)

        if os.path.exists(embeddings_path):
            print(f"Loading existing embeddings from {embeddings_path}")
            parquet_df = pd.read_parquet(embeddings_path).dropna()
            embeddings_df = pd.DataFrame(np.stack(list(parquet_df['embedding']), axis=0))
            embeddings_df.columns = [f"embed_{i}" for i in range(embeddings_df.shape[1])]
            parquet_df = parquet_df.drop('embedding', axis=1)
            embeddings_df = pd.concat([parquet_df.reset_index(drop=True), embeddings_df], axis=1)
            if task_type == TaskType.CLASSIFICATION:
                targets_df = embeddings_df['post_target'].copy()
            elif task_type == TaskType.FORECAST:
                targets_df = embeddings_df['post_forecast_target'].copy()
            elif task_type == TaskType.ANOMALY_DETECTION:
                targets_df = embeddings_df['post_anomaly_target'].copy()
            elif task_type == TaskType.REGRESSION:
                amount_col = sorted([col for col in embeddings_df.columns if 'post_amount' in col])
                assert len(amount_col) > 0
                amount_col = amount_col[0]
                target_values = embeddings_df[amount_col].values
                if hasattr(target_values[0], '__len__'):
                    if 'age' in dataset_name:
                        targets_df = pd.DataFrame([np.log(np.median(v)) for v in target_values], columns=['target'])
                    else:
                        if 'zvuk' in dataset_name:
                            targets_df = pd.DataFrame([np.nanmedian(v) for v in target_values], columns=['target'])
                        else:
                            targets_df = pd.DataFrame([np.log1p(np.nanmedian(v+1e-10)) for v in target_values], columns=['target'])
                else:
                    targets_df = pd.DataFrame(embeddings_df[amount_col]).rename(columns={amount_col: 'target'})
            else:
                raise NotImplimentedError
        else:
            print(f"No embeddings existits {embeddings_path}")
            return
        
        # 3. Split data
        splitter = self.splitter_registry[splitter_name]
        split_data = splitter.split(embeddings_df, targets_df, task_type)

        # 4. Execute downstream task
        task = self.task_registry[task_type]
        results = task.execute(split_data, task_type)

        # 5. Generate report
        report = self._generate_report(dataset_name, embedder_name, splitter_name, task_type, results)

        print("Pipeline completed successfully!")
        return report

    def run_all_configured_experiments(self, use_existing_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Run all experiments configured in task router"""
        pass
    
    def _generate_report(self, dataset_name: str, embedder_name: str, splitter_name: str,
                        task_type: TaskType, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        if task_type == TaskType.CLASSIFICATION:
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_metric = results[best_model]['accuracy']
            metric_name = 'accuracy'
        elif task_type == TaskType.REGRESSION:
            best_model = max(results.keys(), key=lambda x: results[x]['r2'])
            best_metric = results[best_model]['r2']
            metric_name = 'r2'
        elif task_type == TaskType.ANOMALY_DETECTION:
            best_model = max(results.keys(), key=lambda x: results[x]['auc'])
            best_metric = results[best_model]['auc']
            metric_name = 'auc'
        elif task_type == TaskType.FORECAST:
            best_model = max(results.keys(), key=lambda x: results[x]['mse'])
            best_metric = results[best_model]['mse']
            metric_name = 'mse'
        else:
            raise NotImplementedError(f"Task type {task_type} not supported")

        return {
            'dataset': dataset_name,
            'embedder': embedder_name,
            'splitter': splitter_name,
            'task_type': task_type.value,
            'best_model': best_model,
            f'best_{metric_name}': best_metric,
            'all_results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
