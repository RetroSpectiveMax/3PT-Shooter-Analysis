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
