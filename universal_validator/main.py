"""Main execution script with OmegaConf support"""
import sys
import os
import argparse

# Add the current directory to Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

try:
    from universal_validator.config import load_config
    from universal_validator.pipeline.universal_validator import UniversalValidator
    from universal_validator.pipeline.task_router import TaskRouter
    from universal_validator.core.types import TaskType
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the directory that contains universal_validator/")
    sys.exit(1)

def main():
    """Main execution function with CLI support"""
    
    parser = argparse.ArgumentParser(description='Universal Sequence Validator')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (default: universal_validator/config.yaml)')
    parser.add_argument('--dataset', type=str, help='Specific dataset to run')
    parser.add_argument('--task-type', type=str, choices=['classification', 'regression'], help='Specific task type')
    parser.add_argument('--run-all', action='store_true', help='Run all configured experiments')
    parser.add_argument('--use-existing-embeddings', action='store_true', help='Use existing embeddings')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations')
    
    args = parser.parse_args()
    
    try:
        # Load configuration - use default if not specified
        config = load_config(args.config)
        
        # Initialize validator and task router
        validator = UniversalValidator(config)
        task_router = TaskRouter(config)
        
        if args.list_configs:
            task_router.print_available_configurations()
            return
        
        if args.run_all:
            # Run all configured experiments
            reports = validator.run_all_configured_experiments(
                use_existing_embeddings=args.use_existing_embeddings
            )
        else:
            # Run specific experiment
            dataset = args.dataset or 'age'
            task_type = args.task_type or 'classification'
            
            if not task_router.validate_dataset_task(dataset, task_type):
                print(f"Error: Task '{task_type}' not configured for dataset '{dataset}'")
                task_router.print_available_configurations()
                return
            
            report = validator.run_pipeline(
                dataset_name=dataset,
                task_type=TaskType(task_type),
                use_existing_embeddings=args.use_existing_embeddings
            )
            reports = [report]
        
        # Print summary
        print(f"\n{'='*80}")
        print("EXPERIMENT(S) COMPLETED")
        print(f"{'='*80}")
        
        for report in reports:
            task_type = report['task_type']
            metric = 'accuracy' if task_type == 'classification' else 'r2'
            print(f"{report['dataset']} ({task_type}): {report['best_model']} - {metric}: {report[f'best_{metric}']:.4f}")

        return reports
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    reports = main()