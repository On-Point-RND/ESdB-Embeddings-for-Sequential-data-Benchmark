"""Configuration module with OmegaConf support"""
import os
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, List

def load_config(config_path: str = None) -> DictConfig:
    """Load configuration from YAML file"""
    if config_path is None:
        # Config is now in the package directory
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, config_path: str):
    """Save configuration to YAML file"""
    OmegaConf.save(config, config_path)

def merge_with_cli_args(config: DictConfig) -> DictConfig:
    """Merge configuration with command line arguments"""
    cli_conf = OmegaConf.from_cli()
    return OmegaConf.merge(config, cli_conf)

def get_dataset_tasks(config: DictConfig, dataset_name: str) -> List[Dict[str, Any]]:
    """Get configured tasks for a dataset from task router"""
    task_router = config.get('task_router', {})
    dataset_tasks = task_router.get('dataset_tasks', {}).get(dataset_name, [])
    
    if not dataset_tasks:
        # Return default tasks if none specified
        return [
            {
                'type': 'classification',
                'embedder': task_router.get('default_embedder', 'coles'),
                'splitter': task_router.get('default_splitter', 'standard')
            }
        ]
    
    return OmegaConf.to_object(dataset_tasks)

def get_task_config(config: DictConfig, dataset_name: str, task_type: str) -> Dict[str, Any]:
    """Get specific task configuration"""
    tasks = get_dataset_tasks(config, dataset_name)
    for task in tasks:
        if task['type'] == task_type:
            return task
    
    # Return default if not found
    task_router = config.get('task_router', {})
    return {
        'type': task_type,
        'embedder': task_router.get('default_embedder', 'coles'),
        'splitter': task_router.get('default_splitter', 'standard')
    }