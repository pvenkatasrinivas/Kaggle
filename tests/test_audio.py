import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio import wav_duration_sec, wav_durations


def _write_silent_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit samples
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def test_duration_of_two_second_clip(tmp_path):
    clip = tmp_path / "a.wav"
    _write_silent_wav(clip, 2.0)
    assert wav_duration_sec(clip) == pytest.approx(2.0)


def test_durations_keyed_by_filename(tmp_path):
    _write_silent_wav(tmp_path / "x.wav", 1.0)
    _write_silent_wav(tmp_path / "y.wav", 0.5, rate=8000)
    out = wav_durations([tmp_path / "x.wav", tmp_path / "y.wav"])
    assert out == {"x.wav": pytest.approx(1.0), "y.wav": pytest.approx(0.5)}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        wav_duration_sec(tmp_path / "nope.wav")
