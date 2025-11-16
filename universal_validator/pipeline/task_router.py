"""Task router for managing dataset-task configurations"""
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig
from ..core.types import TaskType
from ..config import get_dataset_tasks, get_task_config

class TaskRouter:
    """Routes datasets to appropriate tasks and configurations"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_router_config = config.get('task_router', {})
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        return list(self.config.get('datasets', {}).keys())
    
    def get_dataset_tasks(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get all configured tasks for a dataset"""
        return get_dataset_tasks(self.config, dataset_name)
    
    def get_task_configuration(self, dataset_name: str, task_type: str) -> Dict[str, Any]:
        """Get specific task configuration for a dataset"""
        return get_task_config(self.config, dataset_name, task_type)
    
    def validate_dataset_task(self, dataset_name: str, task_type: str) -> bool:
        """Validate if a task is supported for a dataset"""
        tasks = self.get_dataset_tasks(dataset_name)
        return any(task['type'] == task_type for task in tasks)
    
    def get_default_embedder(self) -> str:
        """Get default embedder"""
        return self.task_router_config.get('default_embedder', 'coles')
    
    def get_default_splitter(self) -> str:
        """Get default splitter"""
        return self.task_router_config.get('default_splitter', 'standard')
    
    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate all experiments from task router configuration"""
        experiments = []
        
        for dataset_name in self.get_available_datasets():
            tasks = self.get_dataset_tasks(dataset_name)
            for task_config in tasks:
                experiments.append({
                    'dataset': dataset_name,
                    'task_type': TaskType(task_config['type']),
                    'embedder': task_config.get('embedder', self.get_default_embedder()),
                    'splitter': task_config.get('splitter', self.get_default_splitter()),
                    'use_existing_embeddings': False
                })
        
        return experiments
    
    def print_available_configurations(self):
        """Print all available dataset-task configurations"""
        print("Available Dataset-Task Configurations:")
        print("=" * 50)
        
        for dataset_name in self.get_available_datasets():
            print(f"\nDataset: {dataset_name}")
            tasks = self.get_dataset_tasks(dataset_name)
            for i, task_config in enumerate(tasks, 1):
                print(f"  {i}. {task_config['type']} "
                      f"(embedder: {task_config.get('embedder', self.get_default_embedder())}, "
                      f"splitter: {task_config.get('splitter', self.get_default_splitter())})")
