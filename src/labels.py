"""Load the training labels and line them up with the feature table rows."""
import csv


def load_labels(csv_path, name_col, label_col="label"):
    """Read a Kaggle train csv into a dict {file name: float label}.

    name_col is the column holding the audio file name; check the competition's
    train.csv header for its exact name. Labels outside 0-5 raise ValueError so a
    wrong column is caught early instead of silently hurting the model.
    """
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = float(row[label_col])
            if not 0 <= value <= 5:
                raise ValueError(f"label {value} for {row[name_col]} is outside 0-5")
            labels[row[name_col]] = value
    return labels


def align_labels(names, labels):
    """Return labels in the same order as names (from feature_table).

    Raises KeyError listing any clip that has no label, so X and y can never
    drift out of order before cross_validate.
    """
    missing = [n for n in names if n not in labels]
    if missing:
        raise KeyError(f"no label for {len(missing)} clip(s), first: {missing[0]}")
    return [labels[n] for n in names]
