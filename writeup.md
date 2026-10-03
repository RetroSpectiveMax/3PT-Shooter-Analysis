# Project Log: Optimal 3-Point Shooting

A running record of what has been done on this project: where the data came from, what was pulled, how it was checked, and what has been built on top of it. The project plan itself lives in `nba_3pt_project.md`.

---

## 1. Environment setup

**Date:** 2026-09-30

- The project root is this folder (`3ptProject`). The plan suggested a nested `nba-3pt-analysis/` folder, but the existing folder was used instead.
- Python 3.12.10, with a virtual environment in `.venv/`.
- Installed packages: `nba_api` (1.11.4), `pandas` (3.0.6), `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `jupyter`, `pyarrow` (25.0.1). Exact versions are pinned in `requirements.txt`.
- Git repository initialized. `.gitignore` excludes `.venv/`, `data/raw/`, `__pycache__/` and `.ipynb_checkpoints/`.

Folder layout:

```
3ptProject/
  data/raw/            # cached API responses, one file per request (gitignored)
  data/raw/shots/      # cached shot-chart responses, one file per team-season
  data/processed/      # combined, cleaned tables
  notebooks/           # 01_q1_optimal_3pa, 02_q2_breakeven
  src/                 # data-pulling scripts
  nba_3pt_project.md   # project plan
  writeup.md           # this log
```

**Location update (2026-10-02):** the project was moved out of OneDrive (it slowed Python startup) to `C:\Users\maxmi\Projects\3ptProject`. Open it in Cursor with File > Open Folder. The move was a verified copy (782 files, identical byte count, identical hashes for all processed parquet files); `.venv/` was rebuilt from `requirements.txt` rather than copied. The original at `C:\Users\maxmi\OneDrive\Desktop\3ptProject` was left untouched and can be deleted once you are happy with the new copy.

---

## 2. Data sources

All data so far comes from **NBA.com's stats API** (`stats.nba.com`), accessed through the free [`nba_api`](https://github.com/swar/nba_api) Python client. No authentication is needed. Every response is saved to `data/raw/` before any processing, and the scripts skip requests whose cached file already exists, so reruns only fetch what's missing.

| Endpoint | Used for | Status |
|---|---|---|
| `LeagueGameLog` | Team box score for every game (Q1) | Pulled, all seasons |
| `ShotChartDetail` | Every field goal attempt with location and result (Q2) | Pulled, all seasons |
| `PlayByPlayV3` | Event-by-event game log (cold-streak analysis) | Tested on one game only |

Basketball-Reference has not been used.

**Coverage:** 12 seasons, 2014-15 through 2025-26, regular season and playoffs. 2014-15 is the first season with shot-level data.

---

## 3. Team game logs (Q1 data)

**Script:** `src/pull_gamelogs.py`
**Raw cache:** `data/raw/gamelog_<season>_<regular_season|playoffs>.parquet` (24 files)
**Output:** `data/processed/team_games.parquet`, 30,878 rows

### What was pulled

One `LeagueGameLog` request per season and season type, with `player_or_team_abbreviation="T"` to get team rows. This gives one row per team per game with the standard box score: FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, plus the matchup, date and win/loss.

Rows per season (each game appears twice, once per team):

| Season | Regular season | Playoffs |
|---|---|---|
| 2014-15 | 2,460 | 162 |
| 2015-16 | 2,460 | 172 |
| 2016-17 | 2,460 | 158 |
| 2017-18 | 2,460 | 164 |
| 2018-19 | 2,460 | 164 |
| 2019-20 | 2,118 | 166 |
| 2020-21 | 2,160 | 170 |
| 2021-22 | 2,460 | 174 |
| 2022-23 | 2,460 | 168 |
| 2023-24 | 2,460 | 164 |
| 2024-25 | 2,460 | 168 |
| 2025-26 | 2,460 | 170 |

2019-20 (COVID stoppage) and 2020-21 (72-game season) have fewer games, as expected.

### Columns added

| Column | Definition |
|---|---|
| `SEASON`, `SEASON_TYPE` | Season label and "Regular Season" / "Playoffs" |
| `HOME` | 1 if home game (matchup contains "vs."), else 0 |
| `WIN` | 1 if `WL == "W"`, else 0 |
| `FG2A`, `FG2M` | `FGA - FG3A` and `FGM - FG3M` |
| `OPP_*` | The opponent's box score for the same game (FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, TEAM_ID) |
| `MARGIN` | `PTS - OPP_PTS` |
| `POSS` | Estimated possessions: `FGA + 0.44*FTA - OREB + TOV`, averaged over both teams |

### Checks

- No missing values in FG3A, FG3M, PTS, OPP_PTS or WL.
- Every `GAME_ID` has exactly two rows, and margins sum to zero within each game.
- League 3PA per team-game rises from 22.4 (2014-15) to 37.6 (2024-25) and 37.0 (2025-26). Estimated pace rises from about 97 to about 102.6 possessions.
- 2018-19 and 2019-20 Rockets average 45.4 and 45.3 3PA per game, matching the Basketball-Reference figures cited in the plan.
- **New finding:** the 2024-25 Celtics averaged 48.2 3PA per game, the highest in the data. They, not the Rockets, are now the high-volume extreme.
- 2026 Finals Game 4 (`GAME_ID 0042500404`, June 10, 2026): Spurs 17-of-43 from three, lost 106-107 to the Knicks, matching the news reports cited in the plan.

---

## 4. Shot-level data (Q2 data)

**Script:** `src/pull_shots.py`
**Raw cache:** `data/raw/shots/<season>_<type>_<team_id>.parquet` (720 files: 12 seasons x 2 season types x 30 teams)
**Output:** `data/processed/shots_<season>.parquet` (12 files, regular season and playoffs combined, told apart by `SEASON_TYPE`)
**Log:** `data/pull_shots.log`

### What was pulled

Every field goal attempt, using `ShotChartDetail` with `context_measure_simple="FGA"`. Columns:

`GAME_ID, GAME_EVENT_ID, PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME, PERIOD, MINUTES_REMAINING, SECONDS_REMAINING, EVENT_TYPE, ACTION_TYPE, SHOT_TYPE, SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_ZONE_RANGE, SHOT_DISTANCE, LOC_X, LOC_Y, SHOT_ATTEMPTED_FLAG, SHOT_MADE_FLAG, GAME_DATE, HTM, VTM`, plus `SEASON` and `SEASON_TYPE`.

### Important issue: league-wide requests are truncated

The plan suggested one league-wide request per season (`team_id=0, player_id=0`). Tested on 2023-24, that request returned exactly 102,400 shots from only 575 games, when a full season has about 230,000 attempts across 1,230 games. The API cuts the result off silently, with no error.

**Fix:** the script requests one team at a time (about 7,000 to 8,000 shots per team-season, well under the cap) and combines the results. The full pull took about 19 minutes. One request hit a dropped connection and succeeded on retry.

### Checks

Shot counts compared with box-score FGA and FG3A totals from the game logs (regular season + playoffs):

| Season | Shots | Box FGA | 3PT shots | Box FG3A |
|---|---|---|---|---|
| 2014-15 | 219,265 | 219,265 | 59,276 | 59,276 |
| 2015-16 | 222,344 | 222,344 | 63,673 | 63,673 |
| 2016-17 | 223,333 | 223,333 | 71,018 | 71,018 |
| 2017-18 | 225,523 | 225,523 | 76,245 | 76,245 |
| 2018-19 | 233,717 | 233,717 | 84,143 | 84,143 |
| 2019-20 | 202,223 | 202,223 | 78,279 | 78,279 |
| 2020-21 | 205,754 | 205,754 | 80,653 | 80,653 |
| 2021-22 | 231,293 | 231,293 | 92,552 | 92,552 |
| 2022-23 | 231,871 | 231,870 | 89,925 | 89,926 |
| 2023-24 | 232,540 | 232,542 | 91,831 | 91,831 |
| 2024-25 | 233,904 | 233,904 | 98,361 | 98,361 |
| 2025-26 | 233,633 | 233,632 | 96,813 | 96,813 |

Total: about 2.7 million shots. Nine seasons match exactly. 2022-23, 2023-24 and 2025-26 are off by one or two shots, which looks like small bookkeeping differences between NBA.com's shot log and its box scores. This is too small to affect any analysis.

---

## 5. Play-by-play (cold-streak data): tested, not yet pulled

- The plan links the older `PlayByPlay` endpoint. The endpoint that works is **`PlayByPlayV3`**.
- Tested on 2026 Finals Game 4. It returned 517 events with columns including `period`, `clock`, `teamTricode`, `playerName`, `shotValue`, `shotResult`, `shotDistance`, `scoreHome`, `scoreAway`, `description` and `actionType`.
- Spurs three-pointers by period from that game: 6-of-10, 8-of-16, 2-of-12, 1-of-5. That is 14-of-26 in the first half and 3-of-17 in the second, matching the reported numbers.
- The endpoint needs one request per game, and each is slow. Expect roughly 30 to 40 minutes for one full season (about 1,300 games including playoffs), and several hours for all 12 seasons.

---

## 6. Data inventory

| File | Rows | Grain | Use |
|---|---|---|---|
| `data/processed/team_games.parquet` | 30,878 | Team x game | Q1 models |
| `data/processed/shots_<season>.parquet` (x12) | about 2.7M total | Shot | Q2 break-even and volume analysis |

Total data on disk (raw + processed): about 67 MB.

---

## 7. Play-by-play pull, 2025-26

**Script:** `src/pull_pbp.py` (resumable; one `PlayByPlayV3` request per game, cached in `data/raw/pbp/`). Run as a background job with log `data/pull_pbp.log`. Combined output: `data/processed/pbp_2025-26.parquet`. **Completed:** 1,315 games (regular season + playoffs), 665,170 events. Checks: every `GAME_ID` in `team_games` is present, and the maximum home plus away score in each game equals the box-score total points for all 1,315 games (the last event row can have blank scores, so use the max). Took about 40 minutes.

---

## 8. Q1 results: optimal team 3PA (`notebooks/01_q1_optimal_3pa.ipynb`)

Models: logistic (win) and OLS (margin) with a squared 3PA term, team-season fixed effects, leave-one-game-out opponent strength, home flag, standard errors clustered by game. "Total" models add nothing else; "full" models add 3P%, 2PA, FTA, TOV, OREB and pace. A boosting model checks the shape without assuming a curve.

Findings:
- **Raw:** win rate by 3PA bucket is flat at about 0.50 (it has to be, since every game has a winner and a loser), but mean margin rises with volume, from -2.5 at 15 or fewer 3PA to +4.8 at 50+. This is mostly good teams shooting more, not proof that threes win.
- **Adjusted (within team-season, controlling for opponent):** the squared term is positive (p about 0.006 logit, 0.013 OLS), so the curve is convex, not an inverted U. Predicted win probability falls as a team shoots more threes than its own norm, flattening above about 45.
- **No interior optimum is identified.** Vertex estimates range from 18 to 155 depending on the model, with confidence intervals that are very wide or span implausible values (for example 41.6 to 89.8 for logit-total). By era the squared term is not significant.
- **Interpretation:** the negative within-team association most likely reflects game state. Teams chase games with threes when trailing, and the box score cannot control for that. This is exactly the confound flagged in the plan. It does not show that taking more threes causes losses. *(Section 11 tested this game-state explanation with 2025-26 play-by-play and did not find support for it.)*
- **Extremes:** 2024-25 Celtics (47.2 3PA per game incl. playoffs, 72% wins) and 2018-19/2019-20 Rockets (45 3PA, 58-63% wins) were winning teams at very high volume.

Limits: no in-game score margin yet (to be added from play-by-play); estimates cover roughly 15 to 55 3PA only.

---

## 9. Q2 results: break-even 3P% (`notebooks/02_q2_breakeven.ipynb`)

League points per 2-point attempt rose from 0.969 (2014-15) to 1.097 (2025-26), so the break-even 3P% rose from 32.3% to 36.6%. League 3P% stayed near 35 to 36.6%. The advantage of the average three over the average two shrank from +0.080 points per attempt in 2014-15 to about zero from 2021-22 on (-0.020 in 2025-26). League volume (3PA share of shots) rose from 27% to 41% in the same period, so teams have taken threes until the average one is roughly worth an average two.

Other findings:
- Corner threes are about 3 points better than above-the-break threes (38.6% vs 35%), stable across seasons.
- Players by 3PA volume per team game: those under 2 shoot 34.0% (below break-even, 41% of them above it) while those taking 6+ shoot 37.8% (80% of them above it). Higher-volume shooters are better shooters, not worse.
- Within-player: a player shoots +0.64 points of 3P% (95% CI +0.51 to +0.76) per extra 3PA per team game, so there is no evidence that more volume hurts a given shooter's accuracy. Selection (coaches give more shots to good shooters) is still possible.

Limits: no contested/open data, only zone and distance; the free-throw adjustment is context only.

---

## 10. Next steps

- Done since this section was written: cold-streak test (section 12) and Q3 roles and lineups (section 13).
- Remaining: README charts, fresh-clone reproducibility test, repo size decision for `data/processed`, publish.

---

## 11. Q1 follow-up: score-margin control (`notebooks/03_q1b_margin_control.ipynb`)

**Housekeeping (2026-10-02):** the OneDrive original was renamed to `C:\Users\maxmi\OneDrive\Desktop\3ptProject_OLD_backup`. It can be deleted once you are satisfied with the new copy.

**Code:** `load_pbp_shots()` in `src/features.py` builds one row per 2025-26 field goal attempt with the shooter's score margin before the shot (233,632 shots, 96,813 threes, matching the box score exactly). Scores are forward-filled because only scoring events carry them. The notebook source is `src/q1b_cells.py`.

Questions were fixed before looking at results: (1) do trailing teams shoot more threes, (2) does the convex curve survive a margin control, (3) does first-half 3PA predict second-half margin once halftime margin is held fixed.

Findings:
- **Trailing teams do not shoot more threes.** The three-point share of shots is 40-43% in every score state (41.2% at trail 1-4, 42.3% at trail 15+, 43.4% at lead 15+). Within the same team-game, +10 points of margin changes the three-point share by +0.16 points (p = 0.34). This weakens the explanation given in section 8, which should now be read as unconfirmed.
- **The convex curve is weak in 2025-26 alone.** Without a margin control the squared term is borderline (margin p = 0.085; win p = 0.19). With a control for the team's average margin at its shots it roughly halves (0.0068 to 0.0030) and is not significant (p = 0.15 margin, 0.57 win). Single-season intervals overlap, so this is not proof that margin explains the curve.
- **No clear optimum.** The vertex ranges from about 30 to 66 across specifications, with wide intervals.
- **Clean-timing design** (first-half 3PA to second-half margin, controlling for halftime margin, first-half average margin and first-half FGA): the linear effect is null (-0.03 points per extra three, 95% CI -0.15 to +0.09). A U-shaped term appears (p = 0.017, minimum near 20 first-half 3PA, tails about +1 point). It is one nominal p-value among several tests, so it is treated as exploratory.

**Conclusion:** Q1 has no clear optimum. The earlier convex pattern is weak and unstable, and the 'game state' explanation did not hold up in the shot data. Caveat: average in-game margin is partly an outcome (correlation 0.87 with final margin), so the margin-controlled models are a robustness check, not a causal estimate.


---

## 12. Cold-streak test (`notebooks/04_cold_streak.ipynb`)

**Code:** `load_threes_with_zone()` in `src/features.py` (play-by-play margin plus shot-chart zone, matched one-to-one on `GAME_ID` and `GAME_EVENT_ID`; all 233,632 shots matched, same shooter in every row). Cell source: `src/cold_cells.py`.

**Pre-specified definitions:** cold = the shooting team's 3P% over its previous N threes in the same game is at most 25% (N = 8, 10, 15); hot = at least 50%; outcome = whether the next three goes in. Controls: shooter fixed effects, shot distance, zone, score margin, home. Game fixed effects are deliberately left out because they create a mechanical negative bias with within-game windows. Standard errors clustered by game.

Findings (96,813 threes, 2025-26):
- **Cold does not predict the next three.** The adjusted effect is -0.4, -0.1 and -0.1 points of make probability for N = 8, 10, 15 (95% CIs roughly -1.2 to +1.0, p from 0.35 to 0.87). Raw 3P% after a cold window is 35.2-35.4% against 35.7-36.1% in the middle group.
- **No hot hand either.** After a hot window the next three is a little less likely to go in (-0.9 to -1.5 points, significant at N = 8 and 10, not at 15). Shuffling outcomes within each team-game shows part of that sign is a finite-sequence artifact (shuffled gaps of about -0.2 to -0.3 points); the excess is about -1 point at N = 8 and about zero at N = 10 and 15.
- Excluding the last five minutes of regulation does not change the result.
- **The Spurs Game 4 pattern is within chance.** Across 2,611 team-games with 8+ threes per half, 73 had a 50%+ first half and 25%-or-worse second half, against about 67 expected from shooter-level chance (95% simulated range 52-83, p = 0.24). First-half and second-half 3P% are uncorrelated (r = -0.003).

**Conclusion:** a null result, which the plan said is a valid finding. Over windows of 8 to 15 threes, a cold stretch does not make the next three worse by more than about a point. Limits: one season, team-level windows only, no defender or openness data.

---

## 13. Q3: roles and lineups (`notebooks/05_q3_roles_lineups.ipynb`)

**Data (`src/pull_roles.py`, cached in `data/raw/roles/`, log `data/pull_roles.log`):** `LeagueDashPlayerStats` (Base and Advanced), `LeagueDashPlayerBioStats` (height) and `LeagueDashLineups` (five-man, Advanced) for 12 regular seasons. `LeagueDashLineups` exists in `nba_api`. Like the shot chart, the league-wide request is **silently cut to 2,000 rows** (Atlanta alone has 508 lineups in 2025-26, but only 65 appear in the league-wide result), so lineups are pulled one team at a time. Total: 185,228 lineups; 12,953 with at least 20 minutes; 12,498 of those have a role for all five players.

**Method (pre-specified):** players with at least 300 minutes (4,871 player-seasons); features z-scored within season (3PA rate, 3PA per 36, usage, assist %, OREB %, DREB %, height, blocks per 36); k-means with k chosen from 4 to 8 by silhouette; lineup net rating regressed on role counts, weighted by possessions, with team-season fixed effects and errors clustered by team-season.

Findings:
- **The clusters are weak.** Silhouette is highest at the smallest k allowed (0.21 at k = 4, falling to 0.18 at k = 8), which means players form a continuum, not distinct types. The four groups are big men (3PA rate 0.08), a high-3PA-rate group of wings and forwards (0.57, called "shooter"), high-usage guards (usage and assists highest, called "creator") and a mixed group of wings. The labels are descriptive.
- **No evidence for diminishing returns.** The squared shooter-count term is positive and not significant (p = 0.24). The simpler count of players with 3PA rate of at least 50% gives the same answer (p = 0.67 for the square). There are very few lineups with 4 or 5 shooters (55 and 1), so the data cannot say anything about extreme mixes.
- **Role differences are modest and depend on the reference.** Replacing a creator with any other role is associated with lower net rating: shooter -0.7 (p = 0.04), big -0.8 (p = 0.10), mixed wing -1.3 (p = 0.001), in points per 100 possessions. This most likely reflects player quality (creators are usually a team's best players) and not the role itself.
- **Era:** the shooter coefficient is about -1.2 in 2014-18 and 2018-22 and about +0.1 in 2022-26 (standard errors about 0.6), so the shooter penalty relative to creators disappears in the latest era. This is descriptive and was not tested formally.

**Conclusion:** there is no clear optimal shooter count and no sign of diminishing returns at observed values, but the analysis cannot separate role from player quality, and role clusters are not well separated. Treat Q3 as exploratory.

---

## 14. Publishing (2026-10-02)

- **App:** `app.py` (Streamlit, plotly). Tabs: findings, league trends, teams (filter by season, team, 3PA, games, win %; choose axes), players (filter by volume, break-even; sortable table), 3PA curve (model overlays with 95% bands), cold streaks (choose N and cold/hot cut-offs; only N = 8, 10, 15 with 25%/50% are marked pre-specified), lineups (filter by role counts, team, player, minutes), break-even calculator, about. It reads `data/app/*.parquet`, built by `src/build_app_data.py`. The cold-streak grid and lineup model in the app reproduce the notebook numbers exactly (for example N = 10: cold -0.146 pp, hot -0.894 pp; shooter coefficient -1.335).
- **Figures:** five charts for the README in `docs/figures/`, made by `src/make_figures.py` from `data/app/`.
- **Repo size:** `data/processed/` (about 67 MB, mostly one 17 MB play-by-play file and twelve 2-2.5 MB shot files) is gitignored and rebuilt by the pull scripts. `data/app/` is under 1 MB and is committed. Git LFS was not needed.
- **Reproducibility test:** fresh clone, new virtual environment, `pip install -r requirements.txt`; results recorded below.
- **Result (2026-10-02):** a fresh `git clone` (42 MB) plus a new virtual environment and `pip install -r requirements.txt` ran the app with no errors (9 tabs) and regenerated all five figures. After copying in `data/processed/` and `data/raw/roles/` (what the pull scripts produce), all five notebooks executed end to end. Notebook 01 needed a longer kernel start-up timeout on the first try (`--ExecutePreprocessor.startup_timeout=300`) because the machine was busy, not because of a code problem.
- **Not done:** pushing to GitHub. There is no remote configured and the GitHub CLI is not installed. No licence has been chosen.
- **Result (2026-10-02):** a fresh `git clone` (42 MB) plus a new virtual environment and `pip install -r requirements.txt` ran the app with no errors (9 tabs) and regenerated all five figures. After copying in `data/processed/` and `data/raw/roles/` (what the pull scripts produce), all five notebooks executed end to end. Notebook 01 needed a longer kernel start-up timeout on the first try (`--ExecutePreprocessor.startup_timeout=300`) because the machine was busy, not because of a code problem.
- **Not done:** pushing to GitHub. There is no remote configured and the GitHub CLI is not installed. No licence has been chosen.
