"""Main execution script with OmegaConf support"""
from pipeline.utils import PipelineConfig
from utils import run_with_config

from universal_validator.pipeline.task_router import TaskRouter
from universal_validator.pipeline.universal_validator import UniversalValidator


def main(cfg: PipelineConfig):
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
        dataset = args.dataset or "age"
        task_type = args.task_type or "classification"

        if not task_router.validate_dataset_task(dataset, task_type):
            print(f"Error: Task '{task_type}' not configured for dataset '{dataset}'")
            task_router.print_available_configurations()
            return

        report = validator.run_pipeline(
            dataset_name=dataset,
            splitter_name=config.task_router.default_splitter,
            task_type=TaskType(task_type),
            use_existing_embeddings=args.use_existing_embeddings,
            embeddings_path=args.parquet_path,
        )
        reports = [report]

    for report in reports:
        task_type = report["task_type"]
        metric = {
            "classification": "accuracy",
            "regression": "r2",
            "anomaly_detection": "auc",
            "forecast": "mse",
        }.get(task_type, "accuracy")

        print(
            f"{report['dataset']} ({task_type}): {report['best_model']} - {metric}: {report[f'best_{metric}']:.4f}"
        )
    print()
    return reports


if __name__ == "__main__":
    try:
        result = run_with_config(main)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
