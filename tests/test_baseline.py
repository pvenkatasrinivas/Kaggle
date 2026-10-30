import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline import baseline_cv


def test_baseline_rmse_close_to_label_std():
    y = np.random.default_rng(0).uniform(0, 5, size=200)
    metrics = baseline_cv(y)
    assert abs(metrics["rmse"] - y.std()) < 0.1
    assert len(metrics["fold_rmse"]) == 5


def test_constant_labels_give_zero_error():
    metrics = baseline_cv([3.0] * 20)
    assert metrics["rmse"] == 0.0
