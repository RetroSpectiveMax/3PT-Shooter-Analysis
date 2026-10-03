"""Build the small summary tables used by the Streamlit app and the README figures.

Reads data/processed (and data/raw/roles for the lineup tables) and writes
data/app/*.parquet. These outputs are small and are committed to git, so the
app and figures work from a fresh clone without re-pulling the NBA data.

Usage: python src/build_app_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    PROCESSED, ROOT, SEASONS, load_pbp_shots, load_shots, load_team_games,
    load_threes_with_zone, vertex_ci,
)

APP = ROOT / "data" / "app"
RAW_ROLES = ROOT / "data" / "raw" / "roles"
CORNER = ["Left Corner 3", "Right Corner 3"]


def save(df: pd.DataFrame, name: str):
    APP.mkdir(parents=True, exist_ok=True)
    df.to_parquet(APP / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)


# ---------------------------------------------------------------- league / zones / players
def league_tables():
    s = load_shots()
    s["IS3"] = s.SHOT_TYPE == "3PT Field Goal"
    s["PTS"] = s.SHOT_MADE_FLAG * np.where(s.IS3, 3, 2)
    g = load_team_games()

    rows = []
    for season, d in s.groupby("SEASON"):
        two, three = d[~d.IS3], d[d.IS3]
        ppa2 = two.PTS.mean()
        rows.append(dict(
            SEASON=season, FG2_PCT=two.SHOT_MADE_FLAG.mean(), PTS_PER_2PA=ppa2,
            FG3_PCT=three.SHOT_MADE_FLAG.mean(), BREAKEVEN_3P=ppa2 / 3,
            EDGE_PER_3PA=3 * three.SHOT_MADE_FLAG.mean() - ppa2,
            SHARE_3PA=len(three) / len(d), N_SHOTS=len(d)))
    seasons = pd.DataFrame(rows)
    pace = g.groupby("SEASON").agg(FG3A_PER_TEAM_GAME=("FG3A", "mean"), POSS=("POSS", "mean")).reset_index()
    seasons = seasons.merge(pace, on="SEASON")
    save(seasons, "seasons")

    z = s[s.IS3 & s.SHOT_ZONE_BASIC.isin(CORNER + ["Above the Break 3"])].copy()
    z["ZONE"] = np.where(z.SHOT_ZONE_BASIC == "Above the Break 3", "Above the break", "Corner")
    zones = z.groupby(["SEASON", "ZONE"]).SHOT_MADE_FLAG.agg(FG3_PCT="mean", FG3A="size").reset_index()
    save(zones, "zones")

    # player-season
    tg = g.groupby(["SEASON", "TEAM_ID"]).size().rename("TEAM_GAMES").reset_index()
    s3 = s[s.IS3].copy()
    s3["CORNER"] = s3.SHOT_ZONE_BASIC.isin(CORNER)
    p = (s3.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID"])
         .agg(FG3A=("SHOT_MADE_FLAG", "size"), FG3M=("SHOT_MADE_FLAG", "sum"), CORNER_3A=("CORNER", "sum"),
              AVG_DIST=("SHOT_DISTANCE", "mean")).reset_index())
    p = p.merge(tg, on=["SEASON", "TEAM_ID"])
    last_team = (s3.sort_values("GAME_DATE").groupby(["SEASON", "PLAYER_ID"]).TEAM_NAME.last()
                 .rename("TEAM").reset_index())
    p = (p.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME"])
         .agg(FG3A=("FG3A", "sum"), FG3M=("FG3M", "sum"), CORNER_3A=("CORNER_3A", "sum"),
              AVG_DIST=("AVG_DIST", "mean"), TEAM_GAMES=("TEAM_GAMES", "max")).reset_index()
         .merge(last_team, on=["SEASON", "PLAYER_ID"]))
    p["FG3_PCT"] = p.FG3M / p.FG3A
    p["FG3A_PER_TEAM_GAME"] = p.FG3A / p.TEAM_GAMES
    p["CORNER_SHARE"] = p.CORNER_3A / p.FG3A
    p = p.merge(seasons[["SEASON", "BREAKEVEN_3P", "PTS_PER_2PA"]], on="SEASON")
    p["EDGE_PER_3PA"] = 3 * p.FG3_PCT - p.PTS_PER_2PA
    p["ABOVE_BREAKEVEN"] = p.FG3_PCT > p.BREAKEVEN_3P
    save(p.drop(columns=["CORNER_3A"]), "players")
    return seasons


# ---------------------------------------------------------------- team-seasons and game table
def team_tables():
    g = load_team_games()
    g["TWO_PCT"] = g.FG2M / g.FG2A
    t = (g.groupby(["SEASON", "TEAM_NAME"])
         .agg(GAMES=("WIN", "size"), WIN_PCT=("WIN", "mean"), MARGIN=("MARGIN", "mean"),
              FG3A=("FG3A", "mean"), FG3M=("FG3M", "sum"), FG3A_TOTAL=("FG3A", "sum"),
              FG2A=("FG2A", "mean"), FTA=("FTA", "mean"), TOV=("TOV", "mean"), POSS=("POSS", "mean"),
              PTS=("PTS", "mean")).reset_index())
    t["FG3_PCT"] = t.FG3M / t.FG3A_TOTAL
    t["FG3A_SHARE"] = t.FG3A / (t.FG3A + t.FG2A)
    t = t.drop(columns=["FG3M", "FG3A_TOTAL"])
    save(t, "teams")

    bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 100]
    g["BUCKET"] = pd.cut(g.FG3A, bins=bins).astype(str)
    b = g.groupby("BUCKET").agg(GAMES=("WIN", "size"), WIN_RATE=("WIN", "mean"), MARGIN=("MARGIN", "mean")).reset_index()
    b["ORDER"] = b.BUCKET.str.extract(r"\((\d+),")[0].astype(int)
    save(b.sort_values("ORDER"), "volume_buckets")


# ---------------------------------------------------------------- Q1 curves
def curves():
    g = load_team_games()
    fe = "C(TEAM_SEASON) + OPP_STRENGTH + HOME"
    xs = np.arange(15, 56)
    out = []

    def add(label, m, a="FG3A", b="FG3A_SQ"):
        p, cov = m.params, np.asarray(m.cov_params())
        idx = [p.index.get_loc(a), p.index.get_loc(b)]
        sub = cov[np.ix_(idx, idx)]
        for x in xs:
            v = np.array([x - 30, x**2 - 900])
            est = p.values[idx] @ v
            se = float(np.sqrt(v @ sub @ v))
            out.append(dict(MODEL=label, FG3A=int(x), EFFECT=est, LO=est - 1.96 * se, HI=est + 1.96 * se))
        est, lo, hi = vertex_ci(p, m.cov_params(), a, b)
        return est, lo, hi

    m_all = smf.ols(f"MARGIN ~ FG3A + FG3A_SQ + {fe}", data=g).fit(cov_type="cluster", cov_kwds={"groups": g.GAME_ID})
    add("2014-26, team-season FE (no margin control)", m_all)

    g25 = g[g.SEASON == "2025-26"].copy()
    s = load_pbp_shots()
    a = s.groupby(["GAME_ID", "TEAM_ID"]).agg(AVG_MARGIN=("MARGIN_BEFORE", "mean")).reset_index()
    g25["GAME_ID"] = g25.GAME_ID.astype(str)
    g25 = g25.merge(a, on=["GAME_ID", "TEAM_ID"])
    m0 = smf.ols(f"MARGIN ~ FG3A + FG3A_SQ + {fe}", data=g25).fit(cov_type="cluster", cov_kwds={"groups": g25.GAME_ID})
    m1 = smf.ols(f"MARGIN ~ FG3A + FG3A_SQ + AVG_MARGIN + {fe}", data=g25).fit(cov_type="cluster", cov_kwds={"groups": g25.GAME_ID})
    add("2025-26 only, no margin control", m0)
    add("2025-26 only, + in-game score margin control", m1)
    save(pd.DataFrame(out), "curves")

    # shot share by score state
    bins = [-80, -15, -10, -5, -1, 0, 4, 9, 14, 80]
    labels = ["trail 15+", "trail 10-14", "trail 5-9", "trail 1-4", "tied", "lead 1-4", "lead 5-9", "lead 10-14", "lead 15+"]
    s["STATE"] = pd.cut(s.MARGIN_BEFORE, bins, labels=labels)
    t = s.groupby("STATE", observed=True).agg(SHOTS=("IS3", "size"), SHARE_3=("IS3", "mean")).reset_index()
    t["STATE"] = t.STATE.astype(str)
    t["ORDER"] = range(len(t))
    save(t, "score_state")


# ---------------------------------------------------------------- cold streak grid
def _within(df, cols, by):
    return df[cols] - df.groupby(by)[cols].transform("mean")


def cold_grid():
    t = load_threes_with_zone()
    t["TG"] = t.GAME_ID + "_" + t.TEAM_ID.astype(str)
    t["M"] = t.MARGIN_BEFORE.clip(-20, 20) / 10
    t["DIST"] = t.SHOT_DISTANCE
    t["corner"] = (t.ZONE == "corner").astype(int)
    t["other"] = (t.ZONE == "other").astype(int)
    rows = []
    for N in [5, 8, 10, 12, 15, 20]:
        t["prior"] = t.groupby("TG").MADE.transform(lambda x: x.shift(1).rolling(N).mean())
        base = t.dropna(subset=["prior"]).copy()
        for cold_cut in [0.20, 0.25, 0.30]:
            for hot_cut in [0.45, 0.50, 0.55]:
                d = base.copy()
                d["cold"] = (d.prior <= cold_cut).astype(int)
                d["hot"] = (d.prior >= hot_cut).astype(int)
                cols = ["MADE", "cold", "hot", "DIST", "corner", "other", "M", "HOME"]
                w = _within(d, cols, "PLAYER_ID")
                m = smf.ols("MADE ~ cold + hot + DIST + corner + other + M + HOME - 1", data=w).fit(
                    cov_type="cluster", cov_kwds={"groups": d.GAME_ID})
                ci = m.conf_int()
                for k, cut in [("cold", cold_cut), ("hot", hot_cut)]:
                    rows.append(dict(
                        N=N, GROUP=k, CUTOFF=cut, SHOTS=int(d[k].sum()), RAW_3P=d.loc[d[k] == 1, "MADE"].mean(),
                        EFFECT_PP=m.params[k] * 100, LO=ci.loc[k, 0] * 100, HI=ci.loc[k, 1] * 100, P=m.pvalues[k],
                        PRESPECIFIED=bool(N in (8, 10, 15) and cold_cut == 0.25 and hot_cut == 0.50)))
        print(f"  cold grid N={N} done", flush=True)
    save(pd.DataFrame(rows), "cold_grid")


# ---------------------------------------------------------------- roles and lineups
def roles_and_lineups():
    def load_players(s):
        b = pd.read_parquet(RAW_ROLES / f"players_base_{s}.parquet")[["PLAYER_ID", "PLAYER_NAME", "MIN", "FGA", "FG3A", "BLK"]]
        a = pd.read_parquet(RAW_ROLES / f"players_adv_{s}.parquet")[["PLAYER_ID", "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT"]]
        h = pd.read_parquet(RAW_ROLES / f"bio_{s}.parquet")[["PLAYER_ID", "PLAYER_HEIGHT_INCHES"]]
        d = b.merge(a, on="PLAYER_ID").merge(h, on="PLAYER_ID", how="left")
        d["SEASON"] = s
        return d

    P = pd.concat([load_players(s) for s in SEASONS], ignore_index=True)
    P = P[P.MIN >= 300].copy()
    P["FG3A_RATE"] = P.FG3A / P.FGA.replace(0, np.nan)
    P["FG3A_36"] = P.FG3A / P.MIN * 36
    P["BLK_36"] = P.BLK / P.MIN * 36
    P["HEIGHT"] = P.PLAYER_HEIGHT_INCHES
    feats = ["FG3A_RATE", "FG3A_36", "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "HEIGHT", "BLK_36"]
    P = P.dropna(subset=feats).copy()
    Z = P.groupby("SEASON")[feats].transform(lambda x: (x - x.mean()) / x.std())
    sil = {}
    for k in range(4, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z).labels_
        sil[k] = silhouette_score(Z, labels, sample_size=3000, random_state=0)
    K = max(sil, key=sil.get)
    km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(Z)
    P["CLUSTER"] = km.labels_
    prof = P.groupby("CLUSTER")[feats].mean()
    names = {prof.FG3A_RATE.idxmax(): "shooter"}
    names[prof.drop(index=list(names)).HEIGHT.idxmax()] = "big"
    names[prof.drop(index=list(names)).AST_PCT.idxmax()] = "creator"
    for c in prof.index:
        names.setdefault(c, "wing")
    P["ROLE"] = P.CLUSTER.map(names)
    save(P[["SEASON", "PLAYER_ID", "PLAYER_NAME", "MIN", "ROLE"] + feats], "player_roles")

    frames = []
    for s in SEASONS:
        for f in sorted(RAW_ROLES.glob(f"lineups_adv_{s}_*.parquet")):
            x = pd.read_parquet(f)
            x["SEASON"] = s
            frames.append(x)
    L = pd.concat(frames, ignore_index=True)
    L = L[L.MIN >= 20].copy()
    role_of = P.set_index(["SEASON", "PLAYER_ID"]).ROLE.to_dict()
    hi3 = (P.set_index(["SEASON", "PLAYER_ID"]).FG3A_RATE >= 0.5).astype(int).to_dict()
    roles, hi, ids = [], [], []
    for season, gid in zip(L.SEASON, L.GROUP_ID):
        pid = [int(i) for i in gid.strip("-").split("-")]
        roles.append([role_of.get((season, i)) for i in pid])
        hi.append([hi3.get((season, i)) for i in pid])
    L["roles"] = roles
    L["hi3"] = hi
    L = L[L.roles.apply(lambda r: all(v is not None for v in r))].copy()
    for r in sorted(set(names.values())):
        L[f"N_{r.upper()}"] = L.roles.apply(lambda x, r=r: sum(v == r for v in x))
    L["N_HI3"] = L.hi3.apply(sum)
    out = L[["SEASON", "TEAM_ABBREVIATION", "GROUP_NAME", "GP", "MIN", "POSS", "OFF_RATING", "DEF_RATING", "NET_RATING",
             "W_PCT", "EFG_PCT", "N_BIG", "N_CREATOR", "N_SHOOTER", "N_WING", "N_HI3"]].copy()
    out["GROUP_NAME"] = out.GROUP_NAME.str.replace(r"\s+-\s+", " / ", regex=True)
    save(out.dropna(subset=["NET_RATING", "POSS"]), "lineups")

    # lineup regression summaries for the app
    L = L.dropna(subset=["NET_RATING", "POSS"]).copy()
    L["TEAM_SEASON"] = L.TEAM_ABBREVIATION + "_" + L.SEASON
    cols = [c for c in L.columns if c.startswith("N_") and c not in ("N_CREATOR", "N_HI3")]
    f = "NET_RATING ~ " + " + ".join(cols) + " + I(N_SHOOTER**2) + C(TEAM_SEASON)"
    m = smf.wls(f, data=L, weights=L.POSS).fit(cov_type="cluster", cov_kwds={"groups": L.TEAM_SEASON})
    keys = [c for c in m.params.index if c.startswith("N_") or c.startswith("I(")]
    ci = m.conf_int()
    save(pd.DataFrame({"TERM": keys, "COEF": m.params[keys].values, "LO": ci.loc[keys, 0].values,
                       "HI": ci.loc[keys, 1].values, "P": m.pvalues[keys].values}), "lineup_model")


if __name__ == "__main__":
    print("league tables"); league_tables()
    print("team tables"); team_tables()
    print("curves"); curves()
    print("roles and lineups"); roles_and_lineups()
    print("cold grid"); cold_grid()
    print("done")
