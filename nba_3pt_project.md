# Optimal 3-Point Shooting: NBA Data Science Project

## 1. Project goal

Figure out how much a team should shoot threes to win, using public NBA data. Three questions:

1. What is the optimal number of team 3-point attempts (3PA) per game?
2. What 3P% must a shooter have, given their volume, for their threes to help the team?
3. What is the optimal mix of 3-point shooters versus other roles on a roster?

Motivation: in the 2026 NBA Finals Game 4, the Spurs made a Finals-record 14 first-half threes (14 of 26) and then made only 3 more, finishing 17-of-43, while blowing a 29-point lead to the Knicks. Sources: [Heavy](https://heavy.com/sports/nba/new-york-knicks/spurs-finals-record-first-half-threes-knicks/), [Sporting News](https://www.sportingnews.com/us/nba/news/what-went-wrong-spurs-blown-finals-lead-knicks/d6353127647e94ab0a34b536).

Caveat: one game is an anecdote, not evidence. Part of that collapse is regression to the mean, and analysts also blamed other factors. Use it as motivation, not as a conclusion.

## 2. Data sources

| Source | What it gives you | Best for |
|---|---|---|
| [nba_api](https://github.com/swar/nba_api) | Free Python client for NBA.com stats, no authentication | Pulling everything programmatically |
| `ShotChartDetail` endpoint | Every shot with coordinates and make/miss flag | Shooter-level 3PA and 3P% |
| `PlayByPlay` endpoint ([docs](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playbyplay.md)) | Event-by-event game log with score and period | Cold-streak and game-state analysis |
| [Basketball-Reference league averages](https://www.basketball-reference.com/leagues/NBA_stats_per_game.html) | Per-game league tables by season | Historical 3-point volume trend |
| [B-Ref team 3PA records](https://www.basketball-reference.com/leaders/team_fg3a.html) | Extreme-volume teams (2018-19 and 2019-20 Rockets, about 45 3PA per game) | Sanity checks at the high end |

Notes:
- Shot-level data is available from 2014-15 onward.
- NBA endpoints can be rate-limited. Add a sleep between requests and cache every response to disk.
- Check Basketball-Reference's terms of use before scraping it heavily. Prefer the API route.
- `LeagueDashLineups` (for lineup-level data) exists in nba_api as far as I know, but verify it in the docs before relying on it.

## 3. Methodology

### Q1: Optimal team 3PA per game
- Unit of analysis: one row per team per game.
- Target: win (logistic regression) or point margin (linear regression / gradient boosting).
- Features: 3PA, 3P%, 2PA, FTA, turnovers, offensive rebounds, pace, home/away, opponent strength.
- Do NOT rely on season averages alone. Good teams shoot more threes partly because they are good.
- Look for a curve: plot win probability against 3PA and find where returns flatten or turn down.

### Q2: Break-even shooter accuracy
A three is worth 3 points, so a shooter breaks even with an average two-point attempt when:

    3P% break-even = (points per 2-point attempt) / 3

If a typical 2PA yields about 1.1 points, the break-even is about 0.367 (36.7%). Compute the real value per season from the data, and include free throws from fouled shots if you want a fuller "points per shot attempt" number.

Then check whether a shooter's efficiency changes as their volume rises (more attempts often means harder attempts). Group shooters into volume buckets and compare 3P% and points per shot across buckets.

### Q3: Optimal mix of shooters vs other roles
- Label roles by clustering players on usage, 3PA rate, rebounding, assists and position (k-means or a Gaussian mixture).
- Build lineup-level data (five-man units) with counts of each role.
- Regress lineup net rating on role counts and look for diminishing returns.
- This is the hardest part. Treat it as a stretch goal.

### Handling "cold" shooting
- Define "cold" before you look at results, for example a team's 3P% over its last N attempts in the game (try N = 8, 10, 15).
- Test whether that state predicts the next shot's outcome. Research on the hot hand suggests the effect is small, so a null result is a legitimate finding.
- Control for shot quality (distance, open vs contested if available) so you don't confuse bad shots with bad luck.
- Control for game state. Teams that trail take more threes, so volume can correlate with losing without causing it.

## 4. Implementation steps

### Step 0: Set up the project

```bash
mkdir nba-3pt-analysis && cd nba-3pt-analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install nba_api pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter pyarrow
git init
mkdir -p data/raw data/processed notebooks src
```

Suggested layout:

```
nba-3pt-analysis/
  data/raw/          # cached API responses (gitignored)
  data/processed/    # cleaned tables
  notebooks/         # exploration and final write-up
  src/               # reusable pulling/cleaning code
  README.md
```

Add `data/raw/` to `.gitignore`.

### Step 1: Pull team game logs (Q1 data)

```python
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2014, 2025)]
frames = []
for s in seasons:
    df = leaguegamelog.LeagueGameLog(
        season=s, season_type_all_star="Regular Season"
    ).get_data_frames()[0]
    df["SEASON"] = s
    df.to_parquet(f"data/raw/gamelog_{s}.parquet")
    frames.append(df)
    time.sleep(1.5)

games = pd.concat(frames, ignore_index=True)
print(games[["TEAM_NAME", "GAME_DATE", "WL", "FG3A", "FG3M", "FG3_PCT"]].head())
```

Check the returned column names and adjust. Consider adding more recent seasons, including 2025-26.

### Step 2: First analysis (team level)

```python
import statsmodels.formula.api as smf

games["WIN"] = (games["WL"] == "W").astype(int)
model = smf.logit("WIN ~ FG3A + FG3_PCT + FG2A_est + TOV + OREB", data=games).fit()
print(model.summary())
```

You will need to derive 2-point attempts as `FGA - FG3A` first. Then try a non-linear term (for example `I(FG3A**2)`) or a binned plot of win rate by 3PA bucket to find the optimal range.

### Step 3: Pull shot-level data (Q2 data)

```python
from nba_api.stats.endpoints import shotchartdetail

shots = shotchartdetail.ShotChartDetail(
    team_id=0, player_id=0,
    season_nullable="2023-24",
    season_type_all_star="Regular Season",
    context_measure_simple="FG3A",
).get_data_frames()[0]
```

Check the `ShotChartDetail` documentation, because pulling league-wide data may require looping over players or teams.

Compute per-player 3PA, 3P% and points per shot, bucket by volume, and compare to the break-even line from section 3.

### Step 4: Play-by-play and cold streaks

Use the `PlayByPlay` endpoint for a sample of games first (for example, all games from one season). Parse the events to build a per-game sequence of 3-point attempts with makes, misses, score margin and time. Then build your "cold" indicator and test whether it predicts the next shot's result or the final margin.

### Step 5: Roles and lineups (stretch)

Pull player season stats, cluster them into roles, pull lineup data, and model net rating against the number of each role on the floor.

### Step 6: Write-up and publishing

- Put a plain-English problem statement at the top of the README.
- Include 3 to 5 clear charts (win probability vs 3PA, break-even line, cold-streak result).
- State your assumptions and limits honestly, including what failed.
- Keep the notebook reproducible: a fresh clone plus `pip install -r requirements.txt` should run it.
- Optional: build a small Streamlit app where a user enters a 3PA and 3P% and sees the predicted win probability.

## 5. Suggested timeline

| Week | Goal |
|---|---|
| 1 | Environment, pull and cache game logs, exploratory plots |
| 2 | Team-level model for Q1, first results |
| 3 | Shot-level data, break-even analysis for Q2 |
| 4 | Play-by-play and cold-streak test |
| 5 | Role clustering and lineup analysis (stretch) |
| 6 | Clean up, README, optional Streamlit app, publish to GitHub |

## 6. Pitfalls checklist

- [ ] Controlled for score margin (trailing teams shoot more threes)
- [x] Didn't treat season averages as game-level evidence
- [ ] Defined "cold" before looking at results
- [ ] Accounted for shot quality where possible
- [x] Reported uncertainty (confidence intervals), not only point estimates
- [x] Cached all API calls and respected rate limits
- [x] Didn't over-generalize from the Spurs Finals game
