"""Write and validate submission.csv for the grammar-scoring competition."""
import pandas as pd

from metrics import clip_scores


def make_submission(ids, preds, path="submission.csv", id_col="filename", target_col="label"):
    """Clip predictions to 0-5 and write a two-column CSV. Returns the DataFrame.

    Check the column names against the competition's sample_submission.csv
    before the first upload and pass id_col / target_col if they differ.
    """
    ids = list(ids)
    preds = clip_scores(preds)
    if len(ids) != len(preds):
        raise ValueError(f"{len(ids)} ids but {len(preds)} predictions")
    df = pd.DataFrame({id_col: ids, target_col: preds})
    df.to_csv(path, index=False)
    return df


def validate_submission(path, expected_ids=None, id_col="filename", target_col="label"):
    """Raise ValueError if the CSV would be rejected by Kaggle; return the DataFrame if OK.

    Checks: exactly the two expected columns, no duplicate or missing ids,
    no NaN scores, every score inside 0-5, and (optionally) that the ids
    match the test set exactly.
    """
    df = pd.read_csv(path)
    if list(df.columns) != [id_col, target_col]:
        raise ValueError(f"columns must be [{id_col}, {target_col}], got {list(df.columns)}")
    if df[id_col].duplicated().any():
        raise ValueError("duplicate ids found")
    if df[target_col].isna().any():
        raise ValueError("NaN scores found")
    if (df[target_col] < 0).any() or (df[target_col] > 5).any():
        raise ValueError("scores must be within 0-5")
    if expected_ids is not None and set(df[id_col]) != set(expected_ids):
        raise ValueError("ids do not match the test set")
    return df
