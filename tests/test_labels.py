import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labels import align_labels, load_labels


def test_load_labels_reads_csv(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("filename,label\na.wav,3.5\nb.wav,1\n", encoding="utf-8")
    assert load_labels(p, "filename") == {"a.wav": 3.5, "b.wav": 1.0}


def test_load_labels_rejects_out_of_range(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("filename,label\na.wav,7\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_labels(p, "filename")


def test_align_labels_follows_name_order():
    assert align_labels(["b.wav", "a.wav"], {"a.wav": 2.0, "b.wav": 4.0}) == [4.0, 2.0]


def test_align_labels_reports_missing():
    with pytest.raises(KeyError):
        align_labels(["a.wav", "z.wav"], {"a.wav": 2.0})
