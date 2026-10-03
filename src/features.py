"""Reusable loading and feature helpers for the 3-point analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2026)]


def load_team_games() -> pd.DataFrame:
    """Team-game table with extra Q1 features.

    Adds:
      TEAM_SEASON    team x season label (for fixed effects)
      OPP_STRENGTH   opponent's mean margin that season, excluding this game
      TEAM_STRENGTH  own mean margin that season, excluding this game
      OPP_FG3_PCT, FG3_PCT (recomputed, in case of NaN)
    """
    g = pd.read_parquet(PROCESSED / "team_games.parquet")
    g["TEAM_ID"] = g["TEAM_ID"].astype(int)
    g["TEAM_SEASON"] = g["TEAM_ABBREVIATION"] + "_" + g["SEASON"]

    grp = g.groupby(["SEASON", "TEAM_ID"])["MARGIN"]
    total, n = grp.transform("sum"), grp.transform("count")
    g["TEAM_STRENGTH"] = (total - g["MARGIN"]) / (n - 1)

    key = g[["GAME_ID", "TEAM_ID", "TEAM_STRENGTH"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "TEAM_STRENGTH": "OPP_STRENGTH"})
    g = g.merge(key, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
    g["FG3_PCT"] = g["FG3M"] / g["FG3A"].replace(0, np.nan)
    g["FG3_PCT"] = g["FG3_PCT"].fillna(g["FG3_PCT"].mean())
    g["FG3A_SQ"] = g["FG3A"] ** 2
    return g


def load_shots(seasons=None) -> pd.DataFrame:
    seasons = seasons or SEASONS
    return pd.concat(
        [pd.read_parquet(PROCESSED / f"shots_{s}.parquet") for s in seasons],
        ignore_index=True,
    )


def vertex_ci(params, cov, a: str, b: str, n_draws=20000, seed=0):
    """Peak of a + b*x + c*x^2 style curve: x* = -b1/(2*b2).

    `a` is the linear term name, `b` the squared term name. Returns
    (estimate, lo, hi) using draws from the coefficient covariance.
    """
    rng = np.random.default_rng(seed)
    idx = [params.index.get_loc(a), params.index.get_loc(b)]
    mu = params.values[idx]
    sub = np.asarray(cov)[np.ix_(idx, idx)]
    d = rng.multivariate_normal(mu, sub, size=n_draws)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = -d[:, 0] / (2 * d[:, 1])
    est = -mu[0] / (2 * mu[1])
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return est, lo, hi


def breakeven_3p(pts_per_2pa: float) -> float:
    """3P% at which a three matches the average two-point attempt."""
    return pts_per_2pa / 3.0


def load_pbp_shots(season: str = "2025-26") -> pd.DataFrame:
    """One row per field goal attempt from play-by-play, with the shooting
    team's score margin *before* the shot.

    Columns added: SEC_ELAPSED (game seconds), MARGIN_BEFORE (shooter's team
    minus opponent, before the shot), IS3, MADE, HOME. Rows keep file order,
    which is chronological; score columns are forward-filled because only
    scoring events carry a score.
    """
    p = pd.read_parquet(PROCESSED / f"pbp_{season}.parquet")
    p["gameId"] = p["gameId"].astype(str)
    p["scoreHome"] = pd.to_numeric(p["scoreHome"], errors="coerce")
    p["scoreAway"] = pd.to_numeric(p["scoreAway"], errors="coerce")
    p["scoreHome"] = p.groupby("gameId")["scoreHome"].ffill().fillna(0)
    p["scoreAway"] = p.groupby("gameId")["scoreAway"].ffill().fillna(0)
    # score before this event = score after the previous event
    p["HOME_BEFORE"] = p.groupby("gameId")["scoreHome"].shift(1).fillna(0)
    p["AWAY_BEFORE"] = p.groupby("gameId")["scoreAway"].shift(1).fillna(0)

    m = p["clock"].str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
    remaining = m[0] * 60 + m[1]
    reg = p["period"].clip(upper=4)
    ot = (p["period"] - 4).clip(lower=0)
    start = np.where(p["period"] <= 4, (p["period"] - 1) * 720, 2880 + (ot - 1) * 300)
    length = np.where(p["period"] <= 4, 720, 300)
    p["SEC_ELAPSED"] = start + (length - remaining)

    s = p[p["isFieldGoal"] == 1].copy()
    tg = pd.read_parquet(PROCESSED / "team_games.parquet")[["GAME_ID", "TEAM_ID", "HOME"]]
    tg["GAME_ID"] = tg["GAME_ID"].astype(str)
    s = s.merge(tg, left_on=["gameId", "teamId"], right_on=["GAME_ID", "TEAM_ID"], how="left")
    own = np.where(s["HOME"] == 1, s["HOME_BEFORE"], s["AWAY_BEFORE"])
    opp = np.where(s["HOME"] == 1, s["AWAY_BEFORE"], s["HOME_BEFORE"])
    s["MARGIN_BEFORE"] = own - opp
    s["IS3"] = (s["shotValue"] == 3).astype(int)
    s["MADE"] = (s["actionType"] == "Made Shot").astype(int)
    return s[["gameId", "TEAM_ID", "period", "SEC_ELAPSED", "MARGIN_BEFORE", "IS3", "MADE",
              "shotDistance", "HOME", "actionNumber", "personId"]].rename(
        columns={"gameId": "GAME_ID", "actionNumber": "GAME_EVENT_ID", "personId": "PLAYER_ID"})


def load_threes_with_zone(season: str = "2025-26") -> pd.DataFrame:
    """Chronological three-point attempts with margin (from play-by-play) and
    shot zone (from the shot chart). Keeps file order, which is time order.

    Adds ZONE: 'corner' / 'above_break' / 'other' (backcourt etc.).
    """
    s = load_pbp_shots(season)
    s = s[s["IS3"] == 1].copy()
    z = pd.read_parquet(PROCESSED / f"shots_{season}.parquet",
                        columns=["GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC", "SHOT_DISTANCE"])
    z["GAME_ID"] = z["GAME_ID"].astype(str)
    s = s.merge(z, on=["GAME_ID", "GAME_EVENT_ID"], how="left", validate="one_to_one")
    s["ZONE"] = np.select(
        [s["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3"]),
         s["SHOT_ZONE_BASIC"] == "Above the Break 3"],
        ["corner", "above_break"], default="other")
    return s
