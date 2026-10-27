import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_table import feature_table


def test_table_aligns_names_and_rows():
    transcripts = {"b.wav": "um hello hello world.", "a.wav": "Good day."}
    durations = {"a.wav": 2.0, "b.wav": 4.0}
    names, columns, rows = feature_table(transcripts, durations)
    assert names == ["a.wav", "b.wav"]
    assert len(rows) == 2 and all(len(r) == len(columns) for r in rows)
    wps = columns.index("words_per_sec")
    assert rows[0][wps] == 1.0
    assert rows[1][wps] == 1.0
    assert rows[1][columns.index("repeated_words")] == 1


def test_skips_clips_missing_a_duration_or_transcript():
    names, _, rows = feature_table({"a.wav": "hi", "c.wav": "yo"}, {"a.wav": 1.0, "b.wav": 1.0})
    assert names == ["a.wav"] and len(rows) == 1


def test_empty_inputs():
    assert feature_table({}, {}) == ([], [], [])
