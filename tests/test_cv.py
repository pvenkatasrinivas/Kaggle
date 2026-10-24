import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv import cross_validate


def _toy_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.clip(2.5 + X @ np.array([0.8, -0.5, 0.3]), 0, 5)
    return X, y


def test_oof_shape_range_and_fit():
    X, y = _toy_data()
    oof, metrics = cross_validate(lambda: Ridge(alpha=0.01), X, y)
    assert oof.shape == (100,)
    assert oof.min() >= 0 and oof.max() <= 5
    assert metrics["rmse"] < 0.3 and metrics["pearson"] > 0.9
    assert len(metrics["fold_rmse"]) == 5


def test_fresh_model_per_fold():
    X, y = _toy_data(n=20)
    built = []
    cross_validate(lambda: built.append(1) or Ridge(), X, y, n_splits=4)
    assert len(built) == 4


def test_length_mismatch_raises():
    X, y = _toy_data(n=10)
    with pytest.raises(ValueError):
        cross_validate(Ridge, X, y[:-1])
