import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcripts import load_cache, save_cache, transcribe_missing


def test_load_missing_file_returns_empty(tmp_path):
    assert load_cache(tmp_path / "transcripts.json") == {}


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "transcripts.json"
    save_cache({"a.wav": "hello there"}, path)
    assert load_cache(path) == {"a.wav": "hello there"}


def test_transcribe_missing_skips_cached_clips(tmp_path):
    path = tmp_path / "transcripts.json"
    save_cache({"a.wav": "cached text"}, path)
    calls = []

    def fake_transcribe(wav):
        calls.append(Path(wav).name)
        return f"text for {Path(wav).name}"

    cache = transcribe_missing(["clips/a.wav", "clips/b.wav"], fake_transcribe, path)
    assert calls == ["b.wav"]  # a.wav was already cached, so Whisper is not called for it
    assert cache == {"a.wav": "cached text", "b.wav": "text for b.wav"}
    assert load_cache(path) == cache  # new transcript was written to disk


def test_second_run_transcribes_nothing(tmp_path):
    path = tmp_path / "transcripts.json"
    transcribe_missing(["x.wav"], lambda w: "once", path)
    assert transcribe_missing(["x.wav"], lambda w: "twice", path) == {"x.wav": "once"}
