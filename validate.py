"""Main execution script with OmegaConf support"""

from universal_validator.pipeline.utils import ValidatorConfig
from universal_validator.utils import run_with_config

from universal_validator.pipeline.universal_validator import UniversalValidator


def main(cfg: ValidatorConfig):
    validator = UniversalValidator(cfg)

    all_tasks = validator.get_available_tasks(verbose=True)
    if cfg.list_configs:
        return

    if cfg.task_names is None:
        tasks = all_tasks
    else:
        assert set(cfg.task_names) <= set(all_tasks)
        tasks = cfg.task_names

    reports = []
    
    for task in tasks:
        report = validator.run_pipeline(task_name=task)
        print(report)
        reports += [report]
    
    return reports


if __name__ == "__main__":
    try:
        result = run_with_config(main, "universal_validator")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
