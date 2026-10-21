"""CV metrics and prediction clipping for the 0-5 grammar score."""
import numpy as np


def clip_scores(preds, low=0.0, high=5.0):
    """Clip predictions into the valid label range (0 to 5)."""
    return np.clip(np.asarray(preds, dtype=float), low, high)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson(y_true, y_pred):
    """Pearson correlation; returns 0.0 if predictions are constant."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def report(y_true, y_pred):
    """Return the three metrics used in the README experiment log.

    Predictions are clipped to 0-5 first, like the final submission.
    """
    y_pred = clip_scores(y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
    }
