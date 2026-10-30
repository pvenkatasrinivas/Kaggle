"""Mean-score baseline: the floor every real model must beat in CV."""
from sklearn.dummy import DummyRegressor

from cv import cross_validate


def baseline_cv(y, n_splits=5, seed=42):
    """Run the same 5-fold CV as a real model, but always predict the train-fold mean.

    Returns the metrics dict from cv.cross_validate. If a model's RMSE is not
    clearly lower than this baseline RMSE, its features are not helping yet.
    Pearson should be close to zero, since the prediction does not use features.
    """
    X = [[0.0]] * len(y)  # features are ignored by the mean predictor
    _, metrics = cross_validate(lambda: DummyRegressor(strategy="mean"), X, y, n_splits, seed)
    return metrics
