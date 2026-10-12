# NBA Three-Point Shooting Project: High-Level Overview

This document explains the whole project in one place: why it was started, what data was collected and how, what analysis was done, what every statistic means, what visualizations exist, what tools were used, and what the results say. For the detailed step-by-step log see `writeup.md`; for the original plan see `nba_3pt_project.md`; for a quick summary see `README.md`.

---

## 1. Purpose

The project started from one anecdote: in Game 4 of the 2026 NBA Finals the Spurs made 14 of their first 26 threes, then only 3 of their last 17, finishing 17-of-43 and losing a big lead to the Knicks. One game proves nothing, so the project asks whether anything general lies behind it:

1. **Q1.** Is there an optimal number of team three-point attempts (3PA) per game?
2. **Q2.** How accurate must a shooter be for his threes to be worth more than an average two-point attempt?
3. **Cold streaks.** Does a team being "cold" from three make its next three less likely to go in (and is there a hot hand)?
4. **Q3.** What mix of shooters and other player types works best in a five-man lineup?

A later addition was a **player shot explorer** (a Buckets-style visualization) so the shot-level data can be looked at directly, one player at a time.

---

## 2. Data: what was gathered and how

**Source.** Everything comes from NBA.com's public stats API, accessed through the free Python package `nba_api`. No login or key is needed. No other website was scraped.

**Coverage.** 12 seasons, 2014-15 through 2025-26 (2014-15 is the first season the API has shot locations). Regular season and playoffs for the game and shot data; regular season only for player and lineup data.

**How it was pulled.** One script per data type in `src/`. Each API response is saved to `data/raw/` before any processing, and scripts skip anything already saved, so a rerun only fetches what is missing. Scripts pause between requests to respect rate limits. A few pulls are slow, and one needed a retry after a dropped connection.

| Data | API endpoint | Script | Size and grain |
|---|---|---|---|
| Team game logs | `LeagueGameLog` | `pull_gamelogs.py` | 30,878 rows, one per team per game (box score) |
| Every shot | `ShotChartDetail` | `pull_shots.py` | about 2.7 million shots, one per field goal attempt, with court location and result |
| Play-by-play (2025-26 only) | `PlayByPlayV3` | `pull_pbp.py` | 1,315 games, 665,170 events |
| Player season stats and heights | `LeagueDashPlayerStats`, `LeagueDashPlayerBioStats` | `pull_roles.py` | about 4,900 qualifying player-seasons |
| Five-man lineups | `LeagueDashLineups` | `pull_roles.py` | 185,228 lineups |

**A problem that shaped the pulls.** The NBA API silently truncates big league-wide requests. A league-wide shot request returned only 102,400 shots (about 45% of a season) with no error, and the league-wide lineup request returned only 2,000 rows. The fix was to request one team at a time (about 7,000 shots per team-season) and combine them.

**Checks that were run.**
- Shot counts match the box-score FGA and 3PA totals exactly in nine seasons and within one or two shots in the other three.
- Every game has exactly two team rows, and the margins sum to zero.
- The maximum home plus away score in each play-by-play game equals the box-score total for all 1,315 games.
- Known facts reproduce: the 2018-19 and 2019-20 Rockets at about 45 3PA per game, and the Spurs' 17-of-43 night (14-of-26 first half, 3-of-17 second half).

**What was derived.** From the raw tables the scripts add home/away, win/loss, two-point attempts and makes, the opponent's box score for the same game, point margin, estimated possessions, the score margin before each shot (from play-by-play), and for the shot explorer a 1-foot court grid with game context.

---

## 3. Analysis

### Q1: Optimal team 3PA (`notebooks/01_q1_optimal_3pa.ipynb`, `03_q1b_margin_control.ipynb`)
- **Unit:** one team in one game.
- **Models:** logistic regression (did the team win) and linear regression (point margin) with a squared 3PA term to allow a hill shape, plus gradient boosting as a check that does not assume a curve shape.
- **Controls:** team-season fixed effects (compare a team only with itself that season, since good teams shoot more threes partly because they are good), opponent strength, home court, and in "full" models 3P%, two-point attempts, free throws, turnovers, offensive rebounds and pace. Standard errors are clustered by game.
- **Follow-up with game state (2025-26):** score margin before every shot, from play-by-play, to test the idea that trailing teams chase games with threes. Also a clean-timing design: first-half 3PA against second-half margin, holding halftime margin fixed.

### Q2: Break-even 3P% (`notebooks/02_q2_breakeven.ipynb`)
- Computes, for each season, the average points per two-point attempt, then the 3P% at which a three yields the same: break-even 3P% = points per 2PA divided by 3.
- Compares league 3P% with that line over time, splits threes by zone (corner vs above the break), and groups shooters by volume to see whether more attempts mean worse accuracy (both between players and within the same player).

### Cold streaks (`notebooks/04_cold_streak.ipynb`)
- Rules fixed before looking at results: "cold" means the team's 3P% over its previous N threes in the same game is 25% or lower (N = 8, 10, 15); "hot" means 50% or higher. The outcome is whether the next three goes in.
- Controls: the shooter, shot distance, zone, score margin and home court. Game fixed effects were left out on purpose because they bias within-game comparisons.
- Extra checks: shuffling outcomes within each game (to measure the artifact that finite sequences create), excluding the last five minutes, and a simulation of how often a Spurs-like first-half/second-half collapse happens by chance.

### Q3: Roles and lineups (`notebooks/05_q3_roles_lineups.ipynb`)
- Players with at least 300 minutes were grouped into roles by k-means clustering on eight standardized stats (three-point rate, 3PA per 36 minutes, usage, assist %, offensive and defensive rebound %, height, blocks per 36). The number of groups (4 to 8) was picked by silhouette score.
- Each five-man lineup's net rating was regressed on how many players of each role it had, weighted by possessions, with team-season fixed effects.

### Shot explorer (added later)
Not a hypothesis test: a descriptive tool. Shots are binned into 1-foot court cells, smoothed with a Gaussian blur, and each spot's FG% is compared with the league average from the same spot, with a shrinkage step so sparse spots do not swing wildly. Details in section 6.

---

## 4. What the categories and stats mean

**Shooting basics**
- **FGA / FGM:** field goal attempts / makes (all shots except free throws).
- **FG3A / FG3M, 3PA / 3PM:** three-point attempts / makes. **3P%** = makes divided by attempts.
- **FG2A / FG2M:** two-point attempts / makes, computed as FGA minus 3PA.
- **FG%:** makes divided by attempts for all field goals.
- **3PA share:** the fraction of all shots that are threes (27% in 2014-15, 41% in 2025-26).

**Break-even analysis**
- **Points per 2PA (PTS_PER_2PA):** league points scored per two-point attempt that season.
- **Break-even 3P% (BREAKEVEN_3P):** points per 2PA divided by 3. A shooter above it produces more points per attempt than an average two.
- **Edge per 3PA (EDGE_PER_3PA):** 3 x 3P% minus points per 2PA. Positive means the average three beats the average two.
- **Above break-even:** the player's 3P% is higher than that season's break-even.
- **Corner vs above the break:** corner threes are from the shortest part of the arc along the sideline; "above the break" is everything else.

**Game and team context**
- **HOME, WIN:** flags for home games and wins. **MARGIN:** team points minus opponent points.
- **POSS (possessions) and pace:** estimated as FGA + 0.44 x FTA - offensive rebounds + turnovers, averaged across both teams.
- **Team-season fixed effects:** the model compares a team with its own average that season, which removes the fact that good teams differ from bad ones.
- **Score margin before the shot:** points ahead or behind when a shot was taken (positive = leading).

**Cold streak terms**
- **Window N:** how many of the team's previous threes in the game are looked at.
- **Cold / hot:** window 3P% at most 25% / at least 50%.
- **Adjusted effect (pp):** the change in the next three's make probability, in percentage points, after controls.
- **95% CI and p-value:** the range the true effect plausibly lies in, and how surprising the result would be if there were no effect. Many tests were run, so a lone p near 0.04 is weak evidence.

**Roles and lineups**
- **Usage %:** share of team plays a player ends (shot, turnover, free throws) while on the floor. **Assist %:** share of teammates' baskets he assists.
- **OREB% / DREB%:** share of available offensive / defensive rebounds he grabs.
- **Role labels:** *Shooter* (high three-point rate, mostly wings and forwards), *Creator* (high-usage guards), *Big* (low three-point rate, tall), *Wing* (mixed). They are descriptive names for statistical clusters.
- **Silhouette score:** how well separated clusters are (1 is perfect, near 0 means they overlap). Here it is about 0.2, so roles are weak.
- **Net rating:** points scored minus allowed per 100 possessions while the lineup is on the floor.

**Shot explorer**
- **Court cell:** a 1 ft by 1 ft square of the half court with the basket at the origin.
- **FG% vs league average (color):** the player's smoothed FG% from that spot minus the league's. Color bands are below -9 points, -9 to -3, -3 to +3, +3 to +9 and above +9.
- **Square size:** how often the player shoots from that spot.
- **Smoothed / Raw / Zones:** smoothed blends nearby cells, raw shows each cell, zones groups the court into ten regions (restricted area, paint, three mid-range spots, two corner threes, three above-the-break spots).
- **Distance curves:** shot frequency and FG% by distance in 1-foot bins, compared with the league and a position group (Guards, Forwards, Bigs; a height-based proxy: 6'5" and under, 6'10" and over).
- **Left vs right:** the same distance curves split by which half of the court the shot came from, with a "difference" line (right minus left).

---

## 5. Tools

| Purpose | Tools |
|---|---|
| Language and environment | Python 3.12, virtual environment, `requirements.txt` with pinned versions, git |
| Data access | `nba_api` (NBA.com stats), `requests` under the hood |
| Data handling | `pandas`, `numpy`, `pyarrow` (Parquet files) |
| Statistics and modeling | `statsmodels` (logistic/OLS, fixed effects, clustered errors), `scikit-learn` (k-means, gradient boosting, silhouette), `scipy` |
| Notebooks | Jupyter (five notebooks, generated from source files in `src/*_cells.py`) |
| Static charts | `matplotlib`, `seaborn` (README figures) |
| Interactive app | Streamlit with Plotly |
| Standalone shot page | plain HTML, CSS and JavaScript with D3.js (loaded from a CDN) |
| Editor | Cursor |

---

## 6. Visualizations

**README charts** (`docs/figures/`, made by `src/make_figures.py`)
1. Break-even 3P% vs league 3P% by season.
2. The adjusted 3PA curve (model-based, with confidence bands).
3. Three-point share by score state.
4. Corner vs above-the-break 3P%.
5. The cold-streak result.

**Streamlit explorer** (`streamlit run app.py`; runs from small committed tables in `data/app/`, so no data pull is needed). Tabs: Findings, League trends, Teams, Players, **Shot charts**, 3PA curve, Cold streaks, Lineups, Break-even calculator, About. Teams, players and lineups can be filtered and sorted; the cold-streak definition can be changed; the calculator takes a shooter's 3P% and gives expected points per three.

**Shot explorer** (the Buckets-style view; a Streamlit tab and a standalone page at `docs/shot_explorer/`). For any player-season with 100+ attempts since 2014-15:
- a half-court map in raw, smoothed or zone mode, with size for frequency and color for accuracy against the league;
- shot frequency and FG% by distance against the league and a position group, with the three-point line marked;
- left-side versus right-side charts of both;
- filters for home/away, wins/losses and opponent, plus sliders for shots per mark and FG% range;
- a two-player compare mode, and PNG export on the standalone page.

Sanity check: James Harden 2015-16 shows 1,617 attempts, 710 makes (43.9%), 657 threes, and a shot-frequency spike just past the arc, matching the Buckets site.

**Shot explorer data** is built by `src/build_shot_chart_data.py` into compact tables (`data/app/shotchart_*.parquet`) and per-season JSON files for the static page. Shared logic (court, smoothing, zones, curves) is in `src/shot_chart.py` and mirrored in `docs/shot_explorer/app.js`.

---

## 7. Conclusions

**1. The average three has caught up to the average two (Q2, high confidence).** League points per two-point attempt rose from 0.969 to 1.097 as twos got more efficient, so the break-even 3P% rose from 32.3% to 36.6%. League 3P% stayed around 35 to 36.6%. The edge of the average three over the average two shrank from +0.080 points per attempt in 2014-15 to about zero from 2021-22 on (-0.020 in 2025-26). Over the same period the share of shots that are threes went from 27% to 41%. In plain terms: teams kept taking more threes until the average one was worth about the same as the average two. This is a description of every shot taken, not a model.

**2. Better shooters take more threes, and more volume does not hurt accuracy (Q2).** Players with under 2 threes per team game shot 34.0% (below break-even); those with 6+ shot 37.8% and 80% of them cleared break-even. Within the same player, an extra three per team game came with +0.64 points of 3P%, so there is no sign that heavier volume lowers a shooter's accuracy, though coaches giving shots to good shooters may explain part of that.

**3. Corner threes are worth about three points of accuracy more (Q2).** 38.6% from the corners against 35.0% above the break, stable across seasons.

**4. There is no clear optimal number of team threes (Q1, inconclusive).** Raw data are misleading: teams that shoot more have better point margins (from -2.5 at 15 or fewer 3PA to +4.8 at 50+), but win rate by volume bucket sits near 50% (necessarily, since every game has a loser) and good teams simply shoot more. With each team compared against itself, the curve is flat or mildly convex, never a hill, and estimates of the "best" volume range from 18 to 155 with huge intervals. The Celtics (about 48 3PA, 72% wins) and Rockets (about 45, 58 to 63% wins) show high volume can go with winning, which is not the same as causing it.

**5. Trailing teams do not chase with threes (Q1 follow-up, one season).** The three-point share of shots is 40 to 43% in every score state, and within a game a 10-point change in margin barely moves it. That weakens the natural explanation for the box-score pattern. Adding a margin control makes the curve weaker still. One caveat: average margin is partly an outcome of the game, so this is a robustness check, not proof of cause.

**6. Cold streaks do not predict the next three, and there is no hot hand (null result, one season).** After a cold stretch (last 8 to 15 team threes at 25% or worse), the next three is within about one percentage point of normal (effects of -0.4, -0.1 and -0.1 points; none distinguishable from zero). After hot stretches there is a small negative effect, partly an artifact of how finite sequences behave, and it fades at larger windows. The Spurs' Game 4 pattern (hot first half, cold second half) is within what chance produces: 73 such team-games were found among 2,611, against about 67 expected. First-half and second-half 3P% are uncorrelated.

**7. Role mixes: no clear optimum, and the data cannot separate role from quality (Q3, exploratory).** Player types form a continuum (silhouette about 0.2), so "shooter" versus "creator" is a loose label. There is no sign of diminishing returns as the number of shooters in a lineup rises (though lineups with 4 or 5 shooters are almost nonexistent). Swapping a creator for another role is associated with a lower net rating, but creators are usually a team's best players, so this mostly reflects talent. The shooter gap that appears in 2014-22 disappears in 2022-26 (descriptive, not formally tested).

**Bottom line.** The data say what the league has already done: the three-point boom pushed the average three to roughly equal value with the average two, which is where a market that is rational on average would end up. They do not support claims of a magic number of threes per game, of a measurable penalty for shooting while cold, or of an ideal number of shooters in a lineup. The Spurs' Game 4 collapse looks like ordinary variance, not a tactical failure of shooting threes.

---

## 8. Limits to keep in mind

- **Observational data.** Teams choose how many threes to take, so associations are not causal. Fixed effects and game-state controls reduce but do not remove that.
- **No defender or openness data.** The API gives location and distance only; shot quality, contest and passing context are missing.
- **Single-season play-by-play.** Margin and cold-streak analyses use 2025-26 only; other seasons would be needed to confirm them.
- **Role labels are statistical clusters** and overlap heavily; they confound role with player quality.
- **Many tests were run.** A single borderline p-value (for example the small hot-hand effect) should be treated as exploratory.
- **Shot explorer approximations.** Locations are binned to 1-foot cells, so distances can be off by a fraction of a foot (three-point totals are exact). Positions are a height proxy. Left/right is as drawn on the chart, not the player's handedness. Only players with 100+ attempts in a season are included. Regular season only.
- **Playoffs** are included in the team-level and shot-level analysis files but excluded from the shot explorer and from player and lineup data.

---

## 9. Where to find things

| Item | Location |
|---|---|
| Quick summary and run instructions | `README.md` |
| Detailed log with numbers and checks | `writeup.md` |
| Original plan | `nba_3pt_project.md` |
| Data pull scripts | `src/pull_*.py` |
| Feature building and notebook sources | `src/features.py`, `src/*_cells.py`, `src/build_notebooks.py`, `src/build_followups.py` |
| Notebooks | `notebooks/01` to `05` |
| Streamlit app and its data | `app.py`, `data/app/`, `src/build_app_data.py` |
| Shot explorer | `src/build_shot_chart_data.py`, `src/shot_chart.py`, `src/shot_chart_ui.py`, `docs/shot_explorer/` |
| README figures | `docs/figures/`, `src/make_figures.py` |
