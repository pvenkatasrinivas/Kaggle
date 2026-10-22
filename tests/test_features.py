import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import text_features


def test_basic_features():
    f = text_features("I I went um to the market. It was good.", 5)
    assert f["word_count"] == 10
    assert f["words_per_sec"] == 2.0
    assert f["filler_count"] == 1
    assert f["repeated_words"] == 1
    assert f["sentence_count"] == 2
    assert f["avg_sentence_len"] == 5.0


def test_empty_transcript():
    f = text_features("", 50)
    assert f["word_count"] == 0 and f["type_token_ratio"] == 0.0
