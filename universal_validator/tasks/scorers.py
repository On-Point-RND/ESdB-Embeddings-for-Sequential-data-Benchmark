from sklearn.metrics import get_scorer

# todo: расширить или стандартизовать до sklearn.metrics.get_scorer_names()?
metric_to_scorer = {
    'accuracy': 'accuracy', 'aucroc': 'roc_auc', 'r2': 'r2',
    'F1': 'f1_weighted', 'precision': 'precision_weighted', 'recall': 'recall_weighted',
    'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'
}


def get_scorers(metric_names):
    """Возвращает список объектов scorer с атрибутом .name, на основе метрики из названия такска"""
    scorers = []
    for name in metric_names:
        try:
            scorer_name = metric_to_scorer[name.lower()]
            scorer = get_scorer(scorer_name)
            scorer.name = scorer_name
            scorers.append(scorer)
        except ValueError as e:
            print(f"Warning: Unsupported metric '{name}': {e}")
    return scorers
