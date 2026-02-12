from sklearn.metrics import get_scorer
from enum import Enum


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# todo: расширить или стандартизовать до sklearn.metrics.get_scorer_names()?
METRIC_INFO = {
    "r2": {"task": TaskType.REGRESSION, "scorer": "r2"},
    "mse": {"task": TaskType.REGRESSION, "scorer": "neg_mean_squared_error"},
    "mae": {"task": TaskType.REGRESSION, "scorer": "neg_mean_absolute_error"},
    "accuracy": {"task": TaskType.CLASSIFICATION, "scorer": "accuracy"},
    "roc_auc": {"task": TaskType.CLASSIFICATION, "scorer": "roc_auc"},
    "f1_weighted": {"task": TaskType.CLASSIFICATION, "scorer": "f1_weighted"},
    "f1_macro": {"task": TaskType.CLASSIFICATION, "scorer": "f1_macro"},
    "f1_micro": {"task": TaskType.CLASSIFICATION, "scorer": "f1_micro"},
    "precision": {"task": TaskType.CLASSIFICATION, "scorer": "precision_weighted"},
    "recall": {"task": TaskType.CLASSIFICATION, "scorer": "recall_weighted"},
}


def get_scorers(metric_names):
    """
    Возвращает список объектов scorer с атрибутом .name, на основе метрики из названия
    ALL SCORERS MUST BE HIGHER == BETTER
    """
    scorers = []
    for name in metric_names:
        try:
            scorer_name = METRIC_INFO[name]["scorer"]
            scorer = get_scorer(scorer_name)
            scorer.name = scorer_name
            scorers.append(scorer)
        except ValueError as e:
            raise ValueError(f"Unsupported metric '{name}': {e}") from e
    return scorers
