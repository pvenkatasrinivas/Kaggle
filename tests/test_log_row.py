import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from log_row import cv_cell, log_row

M = {"rmse": 1.0481, "mae": 0.7859, "pearson": 0.5574}


def test_cv_cell_rounds_to_three_decimals():
    assert cv_cell(M) == "1.048 / 0.786 / 0.557"


def test_log_row_defaults_leaderboard_to_tbd():
    assert log_row(2, "+ audio embeddings", M) == (
        "| 2 | + audio embeddings | 1.048 / 0.786 / 0.557 | TBD |  |"
    )


def test_log_row_with_score_and_notes():
    row = log_row(3, "LightGBM", M, leaderboard="0.7", notes="ok")
    assert row.endswith("| 0.7 | ok |")
