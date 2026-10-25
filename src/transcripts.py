"""Cache Whisper transcripts in transcripts.json so transcription runs only once."""
import json
from pathlib import Path


def load_cache(path="transcripts.json"):
    """Return the {filename: transcript} dict stored at path, or {} if it does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache, path="transcripts.json"):
    """Write the cache to disk as readable JSON (sorted keys, UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def transcribe_missing(wav_paths, transcribe_fn, path="transcripts.json"):
    """Transcribe only the clips not yet in the cache and return the full cache.

    wav_paths: iterable of .wav paths; the cache key is the file name (e.g. "a1.wav").
    transcribe_fn: a function that takes one wav path and returns its transcript,
        for example a thin wrapper around faster-whisper. Passing the function in
        keeps this helper (and its tests) free of any model download.

    The cache is saved after every new clip, so a crash or a Kaggle session
    timeout in the middle of 769 files loses at most one transcript.
    """
    cache = load_cache(path)
    for wav in wav_paths:
        key = Path(wav).name
        if key in cache:
            continue
        cache[key] = transcribe_fn(wav)
        save_cache(cache, path)
    return cache
