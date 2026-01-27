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
    parser.add_argument('--task-type', type=str, choices=['classification', 'regression', 'anomaly_detection', 'forecast'], help='Specific task type')
    parser.add_argument('--run-all', action='store_true', help='Run all configured experiments')
    parser.add_argument('--use-existing-embeddings', action='store_true', help='Use existing embeddings')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations')
    parser.add_argument('-pp','--parquet-path', type=str, default=None, help='Path to cache data directory in parquet format')
    # Splitter configuration overrides
    parser.add_argument('--test-size', type=float, help='Override test size ratio')
    parser.add_argument('--random-state', type=int, help='Override random state')
    
    args = parser.parse_args()
    
    try:
        # Load configuration - use default if not specified
        config = load_config(args.config)
          
        if args.test_size is not None:
            config.splitting.test_size = args.test_size
            print(f"Overriding test_size to: {args.test_size}")
            
        if args.random_state is not None:
            config.splitting.random_state = args.random_state
            print(f"Overriding random_state to: {args.random_state}")
        
        # Initialize validator and task router
        validator = UniversalValidator(config)
             
        if args.run_all:
            # Run all configured experiments
            raise NotImplementedError
        else:
            # Run specific experiment
            dataset = args.dataset or 'age'
            task_type = args.task_type or 'classification'
            
            report = validator.run_pipeline(
                dataset_name=dataset,
                splitter_name="standard",
                task_type=TaskType(task_type),
                use_existing_embeddings=args.use_existing_embeddings,
                embeddings_path=args.parquet_path,
            )
            reports = [report]
        
        for report in reports:
            task_type = report['task_type']
            metric = {
                'classification': 'accuracy',
                'regression': 'r2', 
                'anomaly_detection': 'auc',
                'forecast': 'mse'
            }.get(task_type, 'accuracy')

            print(f"{report['dataset']} ({task_type}): {report['best_model']} - {metric}: {report[f'best_{metric}']:.4f}")
        print()
        return reports
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    reports = main()
