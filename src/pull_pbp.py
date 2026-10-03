"""Pull PlayByPlayV3 for every game of a season (default 2025-26).

One request per game, cached to data/raw/pbp/<game_id>.parquet; cached games
are skipped on rerun, so the job is resumable. Combined output:
data/processed/pbp_<season>.parquet
"""

import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbp"
PROCESSED = ROOT / "data" / "processed"
SLEEP_SECONDS = 0.8
MAX_RETRIES = 5


def fetch_game(game_id: str) -> None:
    path = RAW / f"{game_id}.parquet"
    if path.exists():
        return
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60).get_data_frames()[0]
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"FAILED {game_id}: {exc!r}", flush=True)
                return
            wait = 10 * attempt
            print(f"  {game_id}: {exc!r}, retry in {wait}s", flush=True)
            time.sleep(wait)
    df["gameId"] = game_id
    df.to_parquet(path)
    time.sleep(SLEEP_SECONDS)


def main(season: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(PROCESSED / "team_games.parquet")
    ids = sorted(games.loc[games["SEASON"] == season, "GAME_ID"].astype(str).unique())
    print(f"{season}: {len(ids)} games", flush=True)
    for i, gid in enumerate(ids, 1):
        fetch_game(gid)
        if i % 25 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    missing = [g for g in ids if not (RAW / f"{g}.parquet").exists()]
    if missing:
        print(f"{len(missing)} games missing; rerun to retry: {missing[:10]}", flush=True)
        return
    out = pd.concat([pd.read_parquet(RAW / f"{g}.parquet") for g in ids], ignore_index=True)
    out["SEASON"] = season
    out.to_parquet(PROCESSED / f"pbp_{season}.parquet")
    print(f"done: {len(out)} events", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-26")
