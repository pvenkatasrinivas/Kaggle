# SHL Hiring Assessment 2026: Spoken Grammar Scoring

Kaggle competition: `shl-hiring-assessment-2026`
Kaggle username: `pvenkatasrinivas`

## Problem

Predict a grammar score for each spoken audio clip.
- Input: `.wav` audio files (769 train / 216 test), 45-60 seconds each
- Target: `label`, range 0 to 5
- Metric: see the competition Evaluation page (I report RMSE, MAE and Pearson in my own cross-validation)

## Pipeline

1. **Transcribe first.** Whisper (`small`, via faster-whisper) turns each clip into text, because grammar errors live in the words spoken. Transcripts are cached in `transcripts.json` so this runs only once.
2. **Build features** from each transcript (and later the audio).
3. **Train a model** with 5-fold cross-validation.
4. **Predict test**, clip predictions to 0-5, write `submission.csv`.

## Features

| Group | What it measures |
|---|---|
| Fluency | Words per second, filler words (um, uh), repeated words |
| Text structure | Word count, sentence count, average sentence length, type-token ratio, average word length |
| Text embeddings | Sentence-transformer (all-MiniLM-L6-v2) embedding of the transcript |
| Audio embeddings | Whisper encoder output, mean and std pooled (added in Phase 4) |

## Experiment log (one change per submission)

| # | Change | CV (RMSE / MAE / Pearson) | Leaderboard score | Notes |
|---|---|---|---|---|
| 1 | Baseline: 8 hand features + text embeddings + Ridge | 1.048 / 0.786 / 0.557 | 0.7676 | CV and leaderboard agree, so no overfitting |
| 2 | + Whisper audio embeddings | TBD | TBD | |
| 3 | LightGBM / SVR on combined features | TBD | TBD | |
| 4 | DeBERTa fine-tune (5-fold) | TBD | TBD | |
| 5 | Blend, weights chosen by CV | TBD | TBD | |

## Results

- Final CV score: TBD
- Final leaderboard score: TBD
- CV vs leaderboard gap: TBD

## Repo contents

- `notebook.ipynb`: final Kaggle notebook
- `submission.csv`: final predictions
- `README.md`: this file
- `src/metrics.py`: RMSE, MAE, Pearson and 0-5 clipping used for CV reporting (tests in `tests/`)
- `src/features.py`: fluency and text-structure features from a transcript (tests in `tests/`)
- `src/submission.py`: writes `submission.csv` (clipped to 0-5) and validates it before upload (tests in `tests/`)
- `src/cv.py`: 5-fold cross-validation returning out-of-fold predictions and the RMSE / MAE / Pearson for the experiment log (tests in `tests/`)
- `src/transcripts.py`: `transcripts.json` cache so Whisper transcribes each clip only once, saving after every clip (tests in `tests/`)
- `src/audio.py`: reads each clip's duration from the `.wav` header (standard library only) for the words-per-second feature, keyed by file name like `transcripts.json` (tests in `tests/`)
- `src/feature_table.py`: joins transcripts and clip durations into one feature table (names, columns, rows), skipping clips that lack either (tests in `tests/`)
- `requirements.txt`: Python libraries for the pipeline
- `.gitignore`: keeps audio data, model weights and the Kaggle API token out of git

## What I would try with more time

TBD
