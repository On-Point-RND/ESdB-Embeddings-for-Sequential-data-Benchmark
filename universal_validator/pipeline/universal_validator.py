"""Main pipeline orchestrator with task routing"""
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..core.types import TaskType
from ..core.base_classes import BaseDataset, BaseEmbedder, BaseSplitter, BaseTask
from ..datasets.age_dataset import AgeDataset
from ..embedders.coles_embedder import CoLESEmbedder
from ..splitters.standard_splitter import StandardSplitter
from ..splitters.lastdate_splitter import LastDateSplitter
from ..splitters.client_splitter import ClientSplitter
from ..tasks.classification_task import ClassificationTask
from ..tasks.regression_task import RegressionTask
from .task_router import TaskRouter

class UniversalValidator:
    """Main class that orchestrates the entire validation pipeline with task routing"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.task_router = TaskRouter(config) # seems not yet used properly
        self.dataset_registry = self._initialize_datasets()
        self.embedder_registry = self._initialize_embedders()
        self.splitter_registry = self._initialize_splitters()
        self.task_registry = self._initialize_tasks()

    def _initialize_datasets(self) -> Dict[str, BaseDataset]:
        """Initialize all available datasets"""
        datasets = {}
        dataset_configs = self.config.get('datasets', {})
        
        # Age dataset
        if 'age' in dataset_configs:
            datasets['age'] = AgeDataset(dataset_configs.age)
        
        return datasets

    def _initialize_embedders(self) -> Dict[str, BaseEmbedder]:
        """Initialize all available embedders"""
        return {
            'coles': CoLESEmbedder(self.config.embedding)
        }

    def _initialize_splitters(self) -> Dict[str, BaseSplitter]:
        """Initialize all available data splitters"""
        return {
            'standard': StandardSplitter(self.config.splitting),
            'last_date': LastDateSplitter(self.config.splitting),
            'client': ClientSplitter(self.config.splitting),
            
        }

    def _initialize_tasks(self) -> Dict[TaskType, BaseTask]:
        """Initialize all available tasks"""
        return {
            TaskType.CLASSIFICATION: ClassificationTask(self.config.downstream),
            TaskType.REGRESSION: RegressionTask(self.config.downstream)
        }

    def run_pipeline(self,
                    dataset_name: str = 'age',
                    task_type: TaskType = TaskType.CLASSIFICATION,
                    embedder_name: str = None,
                    splitter_name: str = None,
                    use_existing_embeddings: bool = False,
                    embeddings_path: str = None) -> Dict[str, Any]:
        """Run complete validation pipeline with task routing"""
        
        # Get task configuration from router
        task_config = self.task_router.get_task_configuration(dataset_name, task_type.value)
        
        # Use configured embedder/splitter or defaults
        if embedder_name is None:
            embedder_name = task_config.get('embedder', self.task_router.get_default_embedder())
        if splitter_name is None:
            splitter_name = task_config.get('splitter', self.task_router.get_default_splitter())
        if embeddings_path is None:
            embeddings_path = f'embeddings_{dataset_name}_coles.parquet'

        print(f"Starting {dataset_name} pipeline")
        print(f"  Task: {task_type.value}")
        print(f"  Embedder: {embedder_name}")
        print(f"  Splitter: {splitter_name}")
        print("=" * 60)


        if use_existing_embeddings and os.path.exists(embeddings_path):
            print(f"Loading existing embeddings from {embeddings_path}")
            parquet_df = pd.read_parquet(embeddings_path).dropna()
            embeddings_df = pd.DataFrame(np.stack(list(parquet_df['embedding']), axis=0))
            embeddings_df.columns = [f"embed_{i}" for i in range(embeddings_df.shape[1])]
            parquet_df = parquet_df.drop('embedding', axis=1)
            embeddings_df = pd.concat([parquet_df.reset_index(drop=True), embeddings_df], axis=1)
            if task_type == TaskType.CLASSIFICATION:
                targets_df = embeddings_df['target'].copy()
            elif task_type == TaskType.REGRESSION:
                amount_col = [col for col in embeddings_df.columns if 'amount' in col]
                assert len(amount_col) == 1
                amount_col = amount_col[0]
                target_values = embeddings_df[amount_col].values
                if hasattr(target_values[0], '__len__'):
                    targets_df = pd.DataFrame([np.log(np.median(v)) for v in target_values], columns=['target'])
                else:
                    targets_df = pd.DataFrame(embeddings_df[amount_col]).rename(columns={amount_col: 'target'})
            else:
                raise NotImplimentedError
        else:
            print("Generating new embeddings...")
            # 1. Load and preprocess dataset
            dataset = self.dataset_registry[dataset_name]
            if not dataset.check():
                raise ValueError(f"Dataset {dataset_name} check failed")

            sequences_df, targets_df = dataset.load()
            sequences_df, targets_df = dataset.preprocess(sequences_df, targets_df)

            # 2. Generate or load embeddings
            embedder = self.embedder_registry[embedder_name]
            embeddings_df = embedder.fit_transform(sequences_df)
            # Save embeddings for future use
            embedder.save_embeddings(embeddings_df, embeddings_path)

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
        experiments = self.task_router.generate_experiments()
        all_reports = []
        
        print("Running all configured experiments from task router:")
        self.task_router.print_available_configurations()
        
        for exp_config in experiments:
            try:
                print(f"\n{'='*80}")
                print(f"Experiment: {exp_config['dataset']} - {exp_config['task_type'].value}")
                print(f"Embedder: {exp_config['embedder']}, Splitter: {exp_config['splitter']}")
                print(f"{'='*80}")

                report = self.run_pipeline(
                    dataset_name=exp_config['dataset'],
                    task_type=exp_config['task_type'],
                    embedder_name=exp_config['embedder'],
                    splitter_name=exp_config['splitter'],
                    use_existing_embeddings=use_existing_embeddings
                )
                all_reports.append(report)

            except Exception as e:
                print(f"Experiment failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_reports

    def _generate_report(self, dataset_name: str, embedder_name: str, splitter_name: str,
                        task_type: TaskType, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        if task_type == TaskType.CLASSIFICATION:
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_metric = results[best_model]['accuracy']
            metric_name = 'accuracy'
        else:  # Regression
            best_model = max(results.keys(), key=lambda x: results[x]['r2'])
            best_metric = results[best_model]['r2']
            metric_name = 'r2'

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
