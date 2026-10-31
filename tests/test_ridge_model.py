import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline import baseline_cv
from ridge_model import ridge_cv


def test_ridge_beats_baseline_when_feature_predicts_score():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=200)
    y = 1 + 3 * x + rng.normal(0, 0.1, size=200)
    X = np.column_stack([x * 1000, rng.normal(size=200)])  # one feature on a big scale
    _, metrics = ridge_cv(X, y)
    assert metrics["rmse"] < baseline_cv(y)["rmse"] / 2


def test_predictions_are_clipped_to_0_5():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.linspace(0, 5, 40)
    oof, _ = ridge_cv(X, y)
    assert oof.min() >= 0 and oof.max() <= 5
