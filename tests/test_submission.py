import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from submission import make_submission, validate_submission


def test_make_then_validate_roundtrip(tmp_path):
    path = tmp_path / "submission.csv"
    df = make_submission(["a.wav", "b.wav", "c.wav"], [-0.5, 2.5, 7.0], path)
    assert df["label"].tolist() == [0.0, 2.5, 5.0]
    out = validate_submission(path, expected_ids=["a.wav", "b.wav", "c.wav"])
    assert len(out) == 3


def test_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        make_submission(["a.wav"], [1.0, 2.0], tmp_path / "s.csv")


def test_validate_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("filename,label\na.wav,6\n")
    with pytest.raises(ValueError, match="0-5"):
        validate_submission(bad)
    bad.write_text("filename,label\na.wav,1\na.wav,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(bad)
    bad.write_text("filename,label\na.wav,1\n")
    with pytest.raises(ValueError, match="test set"):
        validate_submission(bad, expected_ids=["a.wav", "b.wav"])
