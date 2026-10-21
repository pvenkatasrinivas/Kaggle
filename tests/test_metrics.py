import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import clip_scores, mae, pearson, report, rmse


def test_clip_scores():
    assert clip_scores([-1, 2.5, 7]).tolist() == [0.0, 2.5, 5.0]


def test_perfect_prediction():
    y = [1, 2, 3, 4]
    assert rmse(y, y) == 0 and mae(y, y) == 0 and abs(pearson(y, y) - 1) < 1e-9


def test_known_values():
    assert abs(rmse([0, 0], [3, 4]) - (12.5 ** 0.5)) < 1e-9
    assert mae([0, 0], [3, 4]) == 3.5


def test_constant_prediction_pearson():
    assert pearson([1, 2, 3], [2, 2, 2]) == 0.0


def test_report_clips_first():
    assert report([5, 5], [9, 9])["rmse"] == 0
