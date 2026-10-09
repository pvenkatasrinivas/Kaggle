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

## What I would try with more time

TBD
