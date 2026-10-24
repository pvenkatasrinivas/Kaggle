"""5-fold cross-validation: out-of-fold predictions plus the README CV metrics."""
import numpy as np
from sklearn.model_selection import KFold

from metrics import clip_scores, report, rmse


def cross_validate(make_model, X, y, n_splits=5, seed=42):
    """Train a fresh model per fold and return (oof_preds, metrics).

    make_model: a function that returns a new, unfitted model with fit/predict,
        for example ``lambda: Ridge(alpha=1.0)``. Building a new model per fold
        guarantees no fold ever trains on its own validation rows.
    X, y: feature matrix and 0-5 labels of the same length.

    oof_preds has one prediction per training row (each made by the model that
    did not see that row), clipped to 0-5 like the final submission.
    metrics is the RMSE / MAE / Pearson dict from metrics.report, which is the
    number to write in the README experiment log, plus "fold_rmse" (one RMSE
    per fold) so you can see how stable the score is across folds.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    if len(X) != len(y):
        raise ValueError(f"{len(X)} rows of X but {len(y)} labels")
    oof = np.zeros(len(y))
    fold_rmse = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(X):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clip_scores(model.predict(X[val_idx]))
        fold_rmse.append(rmse(y[val_idx], oof[val_idx]))
    metrics = report(y, oof)
    metrics["fold_rmse"] = fold_rmse
    return oof, metrics
