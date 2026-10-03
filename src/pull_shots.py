"""Pull shot-level data (every FGA with location and make/miss) for Q2.

League-wide ShotChartDetail requests are silently truncated (~102k rows), so
this loops over teams. Each team-season is cached to
data/raw/shots/<season>_<type>_<team_id>.parquet; cached files are skipped on
rerun. Combined per-season files go to data/processed/shots_<season>.parquet.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "shots"
PROCESSED = ROOT / "data" / "processed"

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2026)]
SEASON_TYPES = ["Regular Season", "Playoffs"]
SLEEP_SECONDS = 1.0
MAX_RETRIES = 4


def fetch_team(season: str, season_type: str, team_id: int) -> pd.DataFrame:
    tag = season_type.lower().replace(" ", "_")
    path = RAW / f"{season}_{tag}_{team_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,
                season_nullable=season,
                season_type_all_star=season_type,
                context_measure_simple="FGA",
                timeout=90,
            ).get_data_frames()[0]
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 10 * attempt
            print(f"  {season} {season_type} {team_id}: {exc!r}, retrying in {wait}s",
                  flush=True)
            time.sleep(wait)

    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type
    df.to_parquet(path)
    time.sleep(SLEEP_SECONDS)
    return df


def main(seasons: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    team_ids = sorted(t["id"] for t in teams.get_teams())
    games = pd.read_parquet(PROCESSED / "team_games.parquet")

    for season in seasons:
        frames = []
        for season_type in SEASON_TYPES:
            for team_id in team_ids:
                frames.append(fetch_team(season, season_type, team_id))
        shots = pd.concat(frames, ignore_index=True)
        shots.to_parquet(PROCESSED / f"shots_{season}.parquet")

        # Shot counts should match box-score FGA/FG3A from the game logs.
        box = games[games["SEASON"] == season][["FGA", "FG3A"]].sum()
        n3 = (shots["SHOT_TYPE"] == "3PT Field Goal").sum()
        print(f"{season}: {len(shots)} shots (box FGA {box.FGA}), "
              f"{n3} threes (box FG3A {box.FG3A})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or SEASONS)
