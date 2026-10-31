"""Ridge regression on the fluency/text features, scored with the shared 5-fold CV."""
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv import cross_validate


def make_ridge(alpha=1.0):
    """Return a new Ridge model that scales features first.

    Features like words-per-second and word count live on very different scales,
    so scaling keeps one large feature from dominating the Ridge penalty.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def ridge_cv(X, y, alpha=1.0, n_splits=5, seed=42):
    """Run 5-fold CV for Ridge and return (oof_preds, metrics) from cv.cross_validate.

    Compare metrics["rmse"] with baseline.baseline_cv(y)["rmse"]: Ridge should
    be clearly lower, otherwise the features are not adding signal yet.
    """
    return cross_validate(lambda: make_ridge(alpha), X, y, n_splits, seed)
