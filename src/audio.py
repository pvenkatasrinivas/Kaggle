"""Read clip durations from .wav headers (needed for the words-per-second feature)."""
import wave
from pathlib import Path


def wav_duration_sec(path):
    """Return the length of one .wav file in seconds.

    Only the header is read (frame count / sample rate), so this takes
    milliseconds per file even for the full 769-clip training set. Uses the
    standard-library ``wave`` module, so no audio library is needed.
    """
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        if rate <= 0:
            raise ValueError(f"{path}: invalid sample rate {rate}")
        return w.getnframes() / rate


def wav_durations(wav_paths):
    """Return {filename: duration_sec} for many clips.

    Keys are bare file names (e.g. "a1.wav"), matching the transcripts.json
    cache in transcripts.py, so the two dicts can be joined on the same key
    when building the feature table.
    """
    return {Path(p).name: wav_duration_sec(p) for p in wav_paths}
