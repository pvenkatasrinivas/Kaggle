"""Hand-made fluency and text-structure features from a transcript."""
import re

FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "hmm"}


def text_features(transcript, duration_sec):
    """Return a dict of simple features for one transcript.

    duration_sec is the audio length in seconds (45-60 in this competition).
    """
    words = re.findall(r"[a-z']+", transcript.lower())
    sentences = [s for s in re.split(r"[.!?]+", transcript) if s.strip()]
    n_words = len(words)
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return {
        "word_count": n_words,
        "words_per_sec": n_words / duration_sec if duration_sec > 0 else 0.0,
        "filler_count": sum(w in FILLERS for w in words),
        "repeated_words": repeats,
        "sentence_count": len(sentences),
        "avg_sentence_len": n_words / len(sentences) if sentences else 0.0,
        "type_token_ratio": len(set(words)) / n_words if n_words else 0.0,
        "avg_word_len": sum(map(len, words)) / n_words if n_words else 0.0,
    }
