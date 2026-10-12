"""Streamlit UI for the Buckets-style player shot charts. Used by app.py ("Shot charts" tab).

Data: data/app/shotchart_*.parquet (built by src/build_shot_chart_data.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import shot_chart as sc  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "app"
HEADSHOT = "https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"


@st.cache_data
def _players() -> pd.DataFrame:
    return pd.read_parquet(DATA / "shotchart_players.parquet")


@st.cache_data
def _meta() -> dict:
    return json.loads((DATA / "shotchart_meta.json").read_text())


@st.cache_data
def _cells(season: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / "shotchart_cells.parquet", filters=[("SEASON", "==", season)])


@st.cache_data
def _baseline(season: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / "shotchart_baseline.parquet", filters=[("SEASON", "==", season)])


def _player_picker(label: str, key: str, season: str, default: str | None = None):
    pl = _players()
    pl = pl[pl.SEASON == season].sort_values("FGA", ascending=False)
    names = pl.PLAYER_NAME.tolist()
    idx = names.index(default) if default in names else 0
    name = st.selectbox(label, names, index=idx, key=key)
    return pl[pl.PLAYER_NAME == name].iloc[0]


def _analysis(row, season, home, result, opp):
    cells = _cells(season)
    cells = cells[cells.PLAYER_ID == row.PLAYER_ID]
    cells = sc.filter_cells(cells, home, result, opp)
    att, made = sc.to_grid(cells)
    base = _baseline(season)
    pos_code = {v: int(k) for k, v in _meta()["positions"].items()}[row.POS]
    lg = sc.to_grid(base[base.pos == -1])
    grp = sc.to_grid(base[base.pos == pos_code])
    t3 = cells[cells.t3 == 1]
    return dict(att=att, made=made, lg=lg, grp=grp, pos=row.POS, a3=int(t3.fga.sum()), m3=int(t3.fgm.sum()))


def _stats_block(row, a):
    fga, fgm = a["att"].sum(), a["made"].sum()
    st.markdown(f"**{row.PLAYER_NAME}**  \n{row.TEAM} | {row.POS} | {row.HEIGHT}")
    c = st.columns(3)
    c[0].metric("FGA", f"{int(fga):,}")
    c[1].metric("FG%", f"{fgm / fga:.1%}" if fga else "-")
    c[2].metric("PTS/G", f"{row.PTS_PG:.1f}" if pd.notna(row.PTS_PG) else "-")
    t_a, t_m = a["a3"], a["m3"]
    c = st.columns(3)
    c[0].metric("3PA", f"{int(t_a):,}")
    c[1].metric("3P%", f"{t_m / t_a:.1%}" if t_a else "-")
    c[2].metric("Games", int(row.GAMES))


def render():
    if not (DATA / "shotchart_cells.parquet").exists():
        st.error("Shot chart data is missing. Run `python src/build_shot_chart_data.py` first (needs data/processed).")
        return
    meta = _meta()
    pl = _players()
    st.subheader("Player shot charts")
    st.caption("Court maps show where a player shoots (square size) and how well, relative to the league average from "
               "the same spot (color). Regular season only. 'Guards / Forwards / Bigs' is a height-based proxy.")

    top = st.columns([1, 1, 1, 1, 1])
    season = top[0].selectbox("Season", sorted(meta["seasons"], reverse=True), key="sc_season")
    compare = top[1].checkbox("Compare two players", False, key="sc_cmp")
    home = top[2].selectbox("Venue", ["All", "Home", "Away"], key="sc_home")
    result = top[3].selectbox("Result", ["All", "Wins", "Losses"], key="sc_res")
    teams = ["All"] + sorted(set(meta["teams"]))
    opp_name = top[4].selectbox("Opponent", teams, key="sc_opp")
    opp = None if opp_name == "All" else meta["teams"].index(opp_name)

    pcols = st.columns(2 if compare else 1)
    with pcols[0]:
        default_a = "James Harden" if season == "2015-16" else None
        row_a = _player_picker("Player", "sc_pa", season, default_a)
    row_b = None
    if compare:
        with pcols[1]:
            pl_s = pl[pl.SEASON == season].sort_values("FGA", ascending=False).PLAYER_NAME.tolist()
            row_b = _player_picker("Compare with", "sc_pb", season, pl_s[1] if len(pl_s) > 1 else None)

    left, right = st.columns([1, 2])
    with left:
        mode = st.radio("Court view", ["Raw", "Smoothed", "Zones"], index=1, horizontal=True, key="sc_mode")
        encode = st.checkbox("Encode shot frequency", True, key="sc_enc")
        legend = st.checkbox("Show legend", True, key="sc_leg")
        min_s, max_s = st.slider("Number of shots (per mark)", 1, 15, (2, 15), key="sc_n",
                                 help="15 means 15 or more. Raw: attempts in a 1-ft cell. Smoothed: shots in the surrounding 2x2 ft.")
        fg_lo, fg_hi = st.slider("Field goal %", 0, 100, (0, 100), key="sc_fg")
        if mode == "Zones":
            st.caption("Zones ignore the shot-count and FG% sliders.")

    players = [r for r in (row_a, row_b) if r is not None]
    analyses = [_analysis(r, season, home, result, opp) for r in players]
    if any(a["att"].sum() == 0 for a in analyses):
        st.warning("No shots match the current filters for at least one player.")
        return

    # court maps
    with right:
        cols = st.columns(len(players))
        for col, r, a in zip(cols, players, analyses):
            with col:
                _stats_block(r, a)
                marks = sc.court_marks(a["att"], a["made"], *a["lg"], mode=mode, min_shots=min_s,
                                       max_shots=1e9 if max_s >= 15 else max_s, fg_lo=fg_lo / 100, fg_hi=fg_hi / 100)
                zones = sc.zone_table(a["att"], a["made"], *a["lg"]) if mode == "Zones" else None
                fig = sc.court_figure(marks, mode, zones, encode, legend, title=f"{r.PLAYER_NAME} {season}",
                                      height=470)
                st.plotly_chart(fig, width="stretch", key=f"court_{r.PLAYER_ID}")

    # distance curves
    st.markdown("---")
    profs = [sc.distance_profile(a["att"], a["made"]) for a in analyses]
    league = sc.distance_profile(*analyses[0]["lg"])
    if compare:
        profiles = {row_a.PLAYER_NAME: profs[0], row_b.PLAYER_NAME: profs[1], "League": league}
    else:
        profiles = {row_a.PLAYER_NAME: profs[0], "League": league,
                    analyses[0]["pos"]: sc.distance_profile(*analyses[0]["grp"])}
    c1, c2 = st.columns(2)
    c1.plotly_chart(sc.frequency_figure(_rename(profiles)), width="stretch", key="sc_freq")
    c2.plotly_chart(sc.fg_figure(_rename(profiles)), width="stretch", key="sc_fgd")

    # left vs right
    for r, p in zip(players, profs):
        if compare:
            st.markdown(f"**{r.PLAYER_NAME}**")
        c1, c2 = st.columns(2)
        c1.plotly_chart(sc.left_right_figure(p, "freq"), width="stretch", key=f"lr_f_{r.PLAYER_ID}")
        c2.plotly_chart(sc.left_right_figure(p, "fg"), width="stretch", key=f"lr_g_{r.PLAYER_ID}")
    st.caption("Left/right is as drawn on the court map (x < 0 is the left half of the chart). "
               "The difference line is right minus left. Distance is measured from the center of 1-ft cells (about +/- 0.5 ft).")


def _rename(profiles: dict) -> dict:
    """The figure helpers color by position: first = player, then others. Keep 'League' under that exact name."""
    return profiles
