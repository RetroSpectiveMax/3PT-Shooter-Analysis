"""Build compact shot-chart tables for the Buckets-style explorer.

Reads data/processed/shots_<season>.parquet, team_games.parquet and data/raw/roles/bio_<season>.parquet.
Writes:
  data/app/shotchart_cells.parquet     player shots binned to 1-ft cells x context (home/win/opponent)
  data/app/shotchart_baseline.parquet  league + position-group baselines, same cell layout (all contexts)
  data/app/shotchart_players.parquet   one row per player-season (team, position proxy, stats)
  data/app/shotchart_meta.json         teams, seasons, grid definition
  docs/shot_explorer/data/<season>.json  columnar JSON per season for the static page (same content)

Court coordinates: NBA LOC_X/LOC_Y are in 1/10 ft, basket at (0, 0). We keep the half court
x in [-25, 25) ft and y in [-5, 47) ft in 1-ft cells: cx = floor(x_ft) + 25 (0..49), cy = floor(y_ft) + 5 (0..51).
Shots beyond the half court are clipped into the last row. Regular season only.

Position proxy (the roles data has no position): height <= 6'5" guard, >= 6'10" big, else forward.

Run: python src/build_shot_chart_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ROLES = ROOT / "data" / "raw" / "roles"
OUT = ROOT / "data" / "app"
WEB = ROOT / "docs" / "shot_explorer" / "data"
MIN_FGA = 100  # drop player-seasons below this from the player tables

NX, NY, X0, Y0 = 50, 52, 25, 5
GUARD_MAX_IN, BIG_MIN_IN = 77, 82
POS_NAMES = {0: "Guards", 1: "Forwards", 2: "Bigs"}


def position_from_height(inches: pd.Series) -> pd.Series:
    pos = pd.Series(1, index=inches.index, dtype="int8")
    pos[inches <= GUARD_MAX_IN] = 0
    pos[inches >= BIG_MIN_IN] = 2
    return pos


def build_season(season: str, tg: pd.DataFrame):
    shots = pd.read_parquet(
        PROC / f"shots_{season}.parquet",
        columns=["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "LOC_X", "LOC_Y", "SHOT_TYPE",
                 "SHOT_MADE_FLAG", "SEASON_TYPE"],
    )
    shots = shots[shots.SEASON_TYPE == "Regular Season"].copy()
    shots["GAME_ID"] = shots.GAME_ID.astype(str).str.zfill(10)
    ctx = tg[(tg.SEASON == season) & (tg.SEASON_TYPE == "Regular Season")][
        ["GAME_ID", "TEAM_ID", "HOME", "WIN", "OPP_TEAM_ID", "TEAM_ABBREVIATION"]
    ].copy()
    ctx["GAME_ID"] = ctx.GAME_ID.astype(str).str.zfill(10)
    shots = shots.merge(ctx, on=["GAME_ID", "TEAM_ID"], how="left")
    missing = shots.HOME.isna().sum()
    if missing:
        print(f"  {season}: {missing} shots without game context (dropped)")
        shots = shots.dropna(subset=["HOME"])

    shots = shots.dropna(subset=["LOC_X", "LOC_Y"])
    x_ft, y_ft = shots.LOC_X / 10.0, shots.LOC_Y / 10.0
    shots["cx"] = (np.floor(x_ft).clip(-X0, X0 - 1) + X0).astype("int16")
    shots["cy"] = (np.floor(y_ft).clip(-Y0, NY - Y0 - 1) + Y0).astype("int16")
    shots["home"] = shots.HOME.astype("int8")
    shots["win"] = shots.WIN.astype("int8")
    shots["opp"] = shots.OPP_TEAM_ID.astype("int64")
    shots["made"] = shots.SHOT_MADE_FLAG.astype("int16")
    shots["t3"] = shots.SHOT_TYPE.str.startswith("3").astype("int8")

    # players: team with most shots, position proxy from height
    bio = pd.read_parquet(ROLES / f"bio_{season}.parquet")
    bio = bio.sort_values("GP", ascending=False).drop_duplicates("PLAYER_ID")
    bio = bio.set_index("PLAYER_ID")
    pos = position_from_height(bio.PLAYER_HEIGHT_INCHES.fillna(79))
    shots["pos"] = shots.PLAYER_ID.map(pos).fillna(1).astype("int8")

    cells = (shots.groupby(["PLAYER_ID", "opp", "home", "win", "cx", "cy", "t3"], as_index=False)
             .agg(fga=("made", "size"), fgm=("made", "sum")))
    cells.insert(0, "SEASON", season)

    base = (shots.groupby(["pos", "cx", "cy"], as_index=False).agg(fga=("made", "size"), fgm=("made", "sum")))
    lg = base.groupby(["cx", "cy"], as_index=False)[["fga", "fgm"]].sum()
    lg.insert(0, "pos", -1)
    base = pd.concat([lg, base], ignore_index=True)
    base.insert(0, "SEASON", season)

    is3 = shots.SHOT_TYPE.str.startswith("3")
    g = shots.groupby("PLAYER_ID")
    team = (shots.groupby(["PLAYER_ID", "TEAM_ABBREVIATION"]).size().reset_index(name="n")
            .sort_values("n", ascending=False).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").TEAM_ABBREVIATION)
    players = pd.DataFrame({
        "PLAYER_NAME": g.PLAYER_NAME.first(),
        "TEAM": team,
        "FGA": g.size(),
        "FGM": g.made.sum(),
        "FG3A": shots[is3].groupby("PLAYER_ID").size(),
        "FG3M": shots[is3].groupby("PLAYER_ID").made.sum(),
    }).fillna({"FG3A": 0, "FG3M": 0})
    players["GAMES"] = g.GAME_ID.nunique()
    players["POS"] = pos.reindex(players.index).fillna(1).astype(int).map(POS_NAMES)
    players["HEIGHT"] = bio.PLAYER_HEIGHT.reindex(players.index)
    players["PTS_PG"] = (bio.PTS / bio.GP.replace(0, np.nan)).reindex(players.index)
    players = players[players.FGA >= MIN_FGA].reset_index()
    players.insert(0, "SEASON", season)
    for c in ("FGA", "FGM", "FG3A", "FG3M", "GAMES"):
        players[c] = players[c].astype(int)
    cells = cells[cells.PLAYER_ID.isin(players.PLAYER_ID)]
    return cells, base, players


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)
    tg = pd.read_parquet(PROC / "team_games.parquet")
    seasons = sorted(p.stem.replace("shots_", "") for p in PROC.glob("shots_*.parquet"))
    latest = tg.sort_values("GAME_DATE").drop_duplicates("TEAM_ID", keep="last")
    teams = {int(r.TEAM_ID): r.TEAM_ABBREVIATION for r in latest.itertuples()}
    team_ids = sorted(teams)
    opp_index = {t: i for i, t in enumerate(team_ids)}

    all_cells, all_base, all_players = [], [], []
    for s in seasons:
        print("season", s)
        c, b, p = build_season(s, tg)
        c["opp"] = c.opp.map(opp_index).astype("int8")
        all_cells.append(c)
        all_base.append(b)
        all_players.append(p)

    cells = pd.concat(all_cells, ignore_index=True)
    base = pd.concat(all_base, ignore_index=True)
    players = pd.concat(all_players, ignore_index=True)
    for df in (cells, base):
        df["fga"] = df.fga.astype("int16")
        df["fgm"] = df.fgm.astype("int16")
    cells["PLAYER_ID"] = cells.PLAYER_ID.astype("int32")
    cells["home"] = cells.home.astype("int8")
    cells["win"] = cells.win.astype("int8")
    cells["t3"] = cells.t3.astype("int8")
    cells["cx"] = cells.cx.astype("int8")
    cells["cy"] = cells.cy.astype("int8")
    base["cx"] = base.cx.astype("int8")
    base["cy"] = base.cy.astype("int8")
    base["pos"] = base.pos.astype("int8")

    cells.to_parquet(OUT / "shotchart_cells.parquet", index=False, compression="zstd")
    base.to_parquet(OUT / "shotchart_baseline.parquet", index=False, compression="zstd")
    players.to_parquet(OUT / "shotchart_players.parquet", index=False)
    meta = {
        "seasons": seasons,
        "teams": [teams[t] for t in team_ids],  # index = opp code
        "grid": {"nx": NX, "ny": NY, "x0": X0, "y0": Y0},
        "positions": {"-1": "League", **{str(k): v for k, v in POS_NAMES.items()}},
    }
    (OUT / "shotchart_meta.json").write_text(json.dumps(meta))
    (WEB / "meta.json").write_text(json.dumps(meta))

    # columnar JSON per season for the static page
    for s in seasons:
        c = cells[cells.SEASON == s]
        b = base[base.SEASON == s]
        p = players[players.SEASON == s]
        obj = {
            "players": {
                "id": p.PLAYER_ID.tolist(), "name": p.PLAYER_NAME.tolist(), "team": p.TEAM.tolist(),
                "pos": p.POS.tolist(), "height": p.HEIGHT.fillna("").tolist(), "pts": p.PTS_PG.round(1).fillna(-1).tolist(),
                "gp": p.GAMES.tolist(), "fga": p.FGA.tolist(), "fgm": p.FGM.tolist(),
                "fg3a": p.FG3A.tolist(), "fg3m": p.FG3M.tolist(),
            },
            # cells sorted by player so the page can slice; hw = home*2 + win
            "cells": {
                "pid": c.PLAYER_ID.tolist(), "opp": c.opp.tolist(), "hw": (c.home * 2 + c.win).tolist(), "t": c.t3.tolist(),
                "cx": c.cx.tolist(), "cy": c.cy.tolist(), "a": c.fga.tolist(), "m": c.fgm.tolist(),
            },
            "base": {
                "pos": b.pos.tolist(), "cx": b.cx.tolist(), "cy": b.cy.tolist(),
                "a": b.fga.tolist(), "m": b.fgm.tolist(),
            },
        }
        (WEB / f"{s}.json").write_text(json.dumps(obj, separators=(",", ":")))
    print("cells", len(cells), "players", len(players))
    for f in sorted(list(OUT.glob("shotchart_*")) + list(WEB.glob("*.json"))):
        print(f"{f.name:32s}{f.stat().st_size / 1e6:7.2f} MB")


if __name__ == "__main__":
    main()
