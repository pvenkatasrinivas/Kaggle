"""Turn the transcripts.json cache and clip durations into a feature table."""
from features import text_features


def feature_table(transcripts, durations):
    """Return (names, columns, rows) for every clip that has a transcript and a duration.

    transcripts and durations are dicts keyed by file name (as in transcripts.json
    and audio.wav_durations). names is the sorted list of clip names, columns the
    feature names, and rows one list of feature values per clip, in the same order
    as names. Clips missing from either dict are skipped so the table never mixes
    in made-up values.
    """
    names = sorted(set(transcripts) & set(durations))
    if not names:
        return [], [], []
    columns = list(text_features("", 1))
    rows = []
    for name in names:
        feats = text_features(transcripts[name], durations[name])
        rows.append([feats[c] for c in columns])
    return names, columns, rows
