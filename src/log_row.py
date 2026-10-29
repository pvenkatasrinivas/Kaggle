"""Format CV metrics as the cell used in the README experiment log."""


def cv_cell(metrics):
    """Return "RMSE / MAE / Pearson" with 3 decimals, e.g. "1.048 / 0.786 / 0.557".

    metrics is the dict from cv.cross_validate (or metrics.report). Copy the
    result into the CV column so every row of the log is written the same way.
    """
    return f"{metrics['rmse']:.3f} / {metrics['mae']:.3f} / {metrics['pearson']:.3f}"


def log_row(number, change, metrics, leaderboard="TBD", notes=""):
    """Return one markdown table row for the experiment log.

    leaderboard stays "TBD" until the Kaggle score is known; it is never guessed.
    """
    return f"| {number} | {change} | {cv_cell(metrics)} | {leaderboard} | {notes} |"
