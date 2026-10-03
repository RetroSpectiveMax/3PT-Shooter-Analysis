# Optimal 3-Point Shooting (NBA)

How many threes should an NBA team take, and how accurate must a shooter be for
his threes to help? This project uses public NBA.com data (via `nba_api`) from
2014-15 to 2025-26 to answer:

1. What is the optimal number of team 3-point attempts per game?
2. What 3P% breaks even against an average two-point attempt?
3. (Stretch) What is the best mix of shooters and other roles?

Motivation: Spurs-Knicks, 2026 Finals Game 4 (17-of-43 from three, a 29-point
lead lost). One game is an anecdote, not evidence.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/pull_gamelogs.py     # team game logs
python src/pull_shots.py        # shot-level data (about 20 min)
python src/pull_pbp.py 2025-26  # play-by-play (about 30-40 min)
```

Raw API responses are cached in `data/raw/` (gitignored); every script skips
cached requests, so reruns only fetch what is missing.

## Layout

- `src/` data pulling and shared helpers (`features.py`)
- `notebooks/01_q1_optimal_3pa.ipynb` Q1 analysis
- `notebooks/02_q2_breakeven.ipynb` Q2 analysis
- `writeup.md` running project log, `nba_3pt_project.md` the plan

Results are summarized in `writeup.md`.
