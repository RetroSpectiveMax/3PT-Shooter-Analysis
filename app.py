"""Interactive explorer for the NBA three-point analysis.

Run: streamlit run app.py
Reads the small summary tables in data/app/ (built by src/build_app_data.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DATA = Path(__file__).parent / "data" / "app"
st.set_page_config(page_title="NBA three-point explorer", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / f"{name}.parquet")


if not (APP_DATA / "seasons.parquet").exists():
    st.error("data/app/ is missing. Run `python src/build_app_data.py` first (needs data/processed).")
    st.stop()

seasons_df, zones, players, teams = load("seasons"), load("zones"), load("players"), load("teams")
SEASON_LIST = sorted(seasons_df.SEASON.unique())

st.title("How many threes, and from whom? An NBA explorer")
st.caption("NBA.com data via nba_api, 2014-15 to 2025-26. Every table here can be filtered and sorted; click a column header to sort.")

tabs = st.tabs(["Findings", "League trends", "Teams", "Players", "3PA curve", "Cold streaks",
                "Lineups", "Break-even calculator", "About"])


def season_range(key: str, default=None):
    lo, hi = st.select_slider("Seasons", options=SEASON_LIST, value=default or (SEASON_LIST[0], SEASON_LIST[-1]), key=key)
    return lo, hi


def in_range(df, lo, hi):
    return df[(df.SEASON >= lo) & (df.SEASON <= hi)]


# ------------------------------------------------------------------ Findings
with tabs[0]:
    st.subheader("What the data say")
    last = seasons_df.iloc[-1]
    c = st.columns(4)
    c[0].metric("League 3P%, 2025-26", f"{last.FG3_PCT:.1%}")
    c[1].metric("Break-even 3P%, 2025-26", f"{last.BREAKEVEN_3P:.1%}",
                help="3P% at which a three equals the average two-point attempt in points.")
    c[2].metric("3PA per team game", f"{last.FG3A_PER_TEAM_GAME:.1f}",
                delta=f"{last.FG3A_PER_TEAM_GAME - seasons_df.iloc[0].FG3A_PER_TEAM_GAME:+.1f} vs 2014-15")
    c[3].metric("Edge of average three vs two", f"{last.EDGE_PER_3PA:+.3f} pts")
    st.markdown("""
| Question | Result | Confidence |
|---|---|---|
| **Q2.** What 3P% breaks even? | Break-even rose from 32.3% (2014-15) to 36.6% (2025-26) as twos got more efficient. League 3P% stayed 35-36.6%, so the average three is now worth about the same as the average two. Corner threes beat above-the-break by about 3 points. | High (description of all shots) |
| **Q1.** Optimal team 3PA? | **No clear optimum.** The adjusted curve is weakly convex in all-season data, weaker and not significant in 2025-26, and the "trailing teams chase with threes" explanation did not hold up in play-by-play. | Inconclusive |
| **Cold streaks** | Being cold over the last 8-15 team threes does not predict the next three (effects within about one point). No hot hand either. The Spurs Game 4 split is within chance. | Null result, one season |
| **Q3.** Best shooter/other mix? | Role clusters are weak; no sign of diminishing returns; role effects are confounded with player quality. | Exploratory |
""")
    st.info("Use the tabs above to filter and sort teams, players and lineups, or to change the cold-streak definition. "
            "Raw associations here are not causal: good teams shoot more threes partly because they are good.")

# ------------------------------------------------------------------ League trends
with tabs[1]:
    st.subheader("League-wide trends")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_scatter(x=seasons_df.SEASON, y=seasons_df.FG3_PCT, name="League 3P%", mode="lines+markers")
        fig.add_scatter(x=seasons_df.SEASON, y=seasons_df.BREAKEVEN_3P, name="Break-even 3P%", mode="lines+markers", line=dict(dash="dash"))
        fig.update_layout(yaxis_tickformat=".0%", title="League 3P% vs break-even", height=380, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.bar(seasons_df, x="SEASON", y="EDGE_PER_3PA", title="Points per three above an average two-point attempt")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.line(zones, x="SEASON", y="FG3_PCT", color="ZONE", markers=True, title="3P% by zone")
        fig.update_layout(yaxis_tickformat=".0%", height=380, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with b:
        metric = st.selectbox("Other league metric", ["FG3A_PER_TEAM_GAME", "SHARE_3PA", "POSS", "FG2_PCT", "PTS_PER_2PA"], key="lm")
        fig = px.line(seasons_df, x="SEASON", y=metric, markers=True, title=metric)
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    st.dataframe(seasons_df.set_index("SEASON").round(3), width="stretch")

# ------------------------------------------------------------------ Teams
with tabs[2]:
    st.subheader("Team seasons")
    lo, hi = season_range("t_seasons")
    t = in_range(teams, lo, hi)
    c = st.columns(4)
    q = c[0].text_input("Team contains", "")
    min3 = c[1].slider("Min 3PA per game", 0.0, 50.0, 0.0, 0.5)
    min_g = c[2].slider("Min games", 0, 100, 30)
    min_w = c[3].slider("Min win %", 0.0, 1.0, 0.0, 0.05)
    if q:
        t = t[t.TEAM_NAME.str.contains(q, case=False)]
    t = t[(t.FG3A >= min3) & (t.GAMES >= min_g) & (t.WIN_PCT >= min_w)]
    st.caption(f"{len(t)} team-seasons (regular season + playoffs)")
    cols = [c for c in t.columns if c not in ("TEAM_NAME", "SEASON")]
    x, y, color = st.columns(3)
    xm = x.selectbox("X axis", cols, index=cols.index("FG3A"))
    ym = y.selectbox("Y axis", cols, index=cols.index("WIN_PCT"))
    cm = color.selectbox("Color", ["SEASON", "none"] + cols)
    fig = px.scatter(t, x=xm, y=ym, color=None if cm == "none" else cm, hover_name="TEAM_NAME",
                     hover_data=["SEASON", "GAMES"], height=480,
                     trendline="ols" if st.checkbox("Show trend line") and cm in ("none", "SEASON") else None)
    st.plotly_chart(fig, width="stretch")
    st.caption("A raw relationship: good teams take more threes partly because they are good. "
               "See the 3PA curve tab for the adjusted analysis.")
    st.dataframe(t.sort_values("WIN_PCT", ascending=False), width="stretch", hide_index=True,
                 column_config={"WIN_PCT": st.column_config.NumberColumn(format="%.3f"),
                                "FG3_PCT": st.column_config.NumberColumn(format="%.3f"),
                                "FG3A_SHARE": st.column_config.NumberColumn(format="%.3f"),
                                "MARGIN": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------------ Players
with tabs[3]:
    st.subheader("Player seasons")
    lo, hi = season_range("p_seasons", (SEASON_LIST[-1], SEASON_LIST[-1]))
    p = in_range(players, lo, hi)
    c = st.columns(4)
    min_a = c[0].slider("Min 3PA (season)", 0, 800, 200, 10)
    min_r = c[1].slider("Min 3PA per team game", 0.0, 12.0, 0.0, 0.5)
    only_above = c[2].checkbox("Only above break-even", False)
    name_q = c[3].text_input("Player contains", "")
    p = p[(p.FG3A >= min_a) & (p.FG3A_PER_TEAM_GAME >= min_r)]
    if only_above:
        p = p[p.ABOVE_BREAKEVEN]
    if name_q:
        p = p[p.PLAYER_NAME.str.contains(name_q, case=False)]
    st.caption(f"{len(p)} player-seasons. Break-even is each season's league figure (in the table: BREAKEVEN_3P).")
    if len(p):
        fig = px.scatter(p, x="FG3A_PER_TEAM_GAME", y="FG3_PCT", size="FG3A", color="EDGE_PER_3PA",
                         color_continuous_scale="RdBu", color_continuous_midpoint=0,
                         hover_name="PLAYER_NAME", hover_data=["SEASON", "TEAM", "FG3A"], height=500)
        be = p.BREAKEVEN_3P.mean()
        fig.add_hline(y=be, line_dash="dash", annotation_text=f"mean break-even {be:.1%}")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="3PA per team game", yaxis_title="3P%")
        st.plotly_chart(fig, width="stretch")
        pshow = p[["PLAYER_NAME", "SEASON", "TEAM", "FG3A", "FG3M", "FG3_PCT", "FG3A_PER_TEAM_GAME", "BREAKEVEN_3P",
                   "EDGE_PER_3PA", "CORNER_SHARE", "AVG_DIST"]]
        st.dataframe(pshow.sort_values("EDGE_PER_3PA", ascending=False), width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["FG3_PCT", "FG3A_PER_TEAM_GAME", "BREAKEVEN_3P", "EDGE_PER_3PA", "CORNER_SHARE", "AVG_DIST"]})
        st.caption("EDGE_PER_3PA = 3 x 3P% minus the league's points per two-point attempt that season. "
                   "High-volume shooters are mostly better shooters, so volume alone does not lower efficiency.")

# ------------------------------------------------------------------ 3PA curve
with tabs[4]:
    st.subheader("Q1: Is there an optimal number of team three-point attempts?")
    curves, buckets, state = load("curves"), load("volume_buckets"), load("score_state")
    models = st.multiselect("Models", sorted(curves.MODEL.unique()), default=sorted(curves.MODEL.unique()))
    fig = go.Figure()
    for m in models:
        d = curves[curves.MODEL == m]
        fig.add_scatter(x=d.FG3A, y=d.EFFECT, name=m, mode="lines")
        fig.add_scatter(x=pd.concat([d.FG3A, d.FG3A[::-1]]), y=pd.concat([d.HI, d.LO[::-1]]), fill="toself",
                        opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(xaxis_title="team 3PA per game", yaxis_title="predicted margin vs 30 3PA (points)", height=480,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.markdown("""
Shaded bands are 95% intervals. A real optimum would show as a hill; instead the curves are flat or slightly U-shaped, and the
2025-26 intervals include zero everywhere. **Conclusion: the data do not identify a best number of threes.**
Estimates cover about 15-55 attempts only.""")
    a, b = st.columns(2)
    with a:
        fig = px.bar(buckets, x="BUCKET", y="MARGIN", title="Raw: mean margin by 3PA bucket (confounded)")
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.line(state.sort_values("ORDER"), x="STATE", y="SHARE_3", markers=True,
                      title="Share of shots that are threes, by score state (2025-26)")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0.3, 0.5])
        st.plotly_chart(fig, width="stretch")
        st.caption("Trailing teams do not shoot more threes, so 'chasing the game' does not explain the curve.")

# ------------------------------------------------------------------ Cold streaks
with tabs[5]:
    st.subheader("Does a cold stretch predict the next three? (2025-26)")
    grid = load("cold_grid")
    c = st.columns(3)
    N = c[0].select_slider("Window N (previous team threes in the game)", sorted(grid.N.unique()), value=10)
    cold_cut = c[1].select_slider("Cold = prior 3P% at most", sorted(grid[grid.GROUP == "cold"].CUTOFF.unique()), value=0.25)
    hot_cut = c[2].select_slider("Hot = prior 3P% at least", sorted(grid[grid.GROUP == "hot"].CUTOFF.unique()), value=0.5)
    sel = grid[(grid.N == N) & (((grid.GROUP == "cold") & (grid.CUTOFF == cold_cut)) | ((grid.GROUP == "hot") & (grid.CUTOFF == hot_cut)))]
    pre = bool(sel.PRESPECIFIED.all())
    st.markdown("**Pre-specified analysis**" if pre else
                "**Exploratory setting.** Only N = 8, 10, 15 with cold <= 25% and hot >= 50% were fixed in advance; "
                "other settings are for exploration and every extra setting raises the chance of a false positive.")
    for _, r in sel.iterrows():
        st.metric(f"After a {r.GROUP} stretch ({r.SHOTS:,} shots)", f"{r.EFFECT_PP:+.2f} pp",
                  help=f"95% CI {r.LO:+.2f} to {r.HI:+.2f}, p = {r.P:.3f}. Raw 3P% in this group {r.RAW_3P:.1%}.")
        st.caption(f"{r.GROUP}: adjusted effect on next-three make probability, 95% CI {r.LO:+.2f} to {r.HI:+.2f} pp, p = {r.P:.3f}")
    show = grid[(grid.CUTOFF == np.where(grid.GROUP == "cold", cold_cut, hot_cut))]
    fig = go.Figure()
    for grp, col in [("cold", "royalblue"), ("hot", "crimson")]:
        d = grid[(grid.GROUP == grp) & (grid.CUTOFF == (cold_cut if grp == "cold" else hot_cut))].sort_values("N")
        fig.add_scatter(x=d.N, y=d.EFFECT_PP, name=grp, mode="markers+lines", marker_color=col,
                        error_y=dict(type="data", symmetric=False, array=d.HI - d.EFFECT_PP, arrayminus=d.EFFECT_PP - d.LO))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(xaxis_title="window N", yaxis_title="effect on next-three make prob (pp)", height=420,
                      title="Adjusted effect across window sizes (95% CI)")
    st.plotly_chart(fig, width="stretch")
    st.markdown("""
Controls: shooter fixed effects, shot distance, zone, score margin, home. **Conclusion: a null.** Cold stretches do not make the
next three worse by more than about a point. The small negative after hot stretches is partly a finite-sequence artifact
(see the notebook), not a streak effect. The Spurs Game 4 pattern (hot first half, cold second half) occurred in 73 of 2,611
team-games against about 67 expected by chance.""")

# ------------------------------------------------------------------ Lineups
with tabs[6]:
    st.subheader("Q3: Lineups by role mix (exploratory)")
    L, lm = load("lineups"), load("lineup_model")
    lo, hi = season_range("l_seasons", (SEASON_LIST[-4], SEASON_LIST[-1]))
    L = in_range(L, lo, hi)
    c = st.columns(4)
    min_min = c[0].slider("Min minutes together", 20, 400, 50, 5)
    team = c[1].multiselect("Team", sorted(L.TEAM_ABBREVIATION.unique()))
    pname = c[2].text_input("Player in lineup", "")
    c3 = c[3].selectbox("Color by", ["N_SHOOTER", "N_BIG", "N_CREATOR", "N_WING", "N_HI3"])
    r = st.columns(4)
    rng = {k: r[i].slider(f"{k.replace('N_', '').title()}s", 0, 5, (0, 5)) for i, k in
           enumerate(["N_SHOOTER", "N_BIG", "N_CREATOR", "N_WING"])}
    d = L[L.MIN >= min_min]
    for k, (a, b) in rng.items():
        d = d[(d[k] >= a) & (d[k] <= b)]
    if team:
        d = d[d.TEAM_ABBREVIATION.isin(team)]
    if pname:
        d = d[d.GROUP_NAME.str.contains(pname, case=False)]
    st.caption(f"{len(d):,} lineups. Role labels come from k-means on player stats and are descriptive; clusters overlap a lot.")
    if len(d):
        fig = px.box(d, x="N_SHOOTER", y="NET_RATING", points="outliers", title="Net rating by number of 'shooter' players")
        st.plotly_chart(fig, width="stretch")
        fig = px.scatter(d, x="POSS", y="NET_RATING", color=d[c3].astype(str), hover_name="GROUP_NAME",
                         hover_data=["SEASON", "TEAM_ABBREVIATION", "MIN"], height=450, log_x=True,
                         title="Each lineup: net rating vs possessions (noisy at small samples)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(d.sort_values("NET_RATING", ascending=False), width="stretch", hide_index=True)
    st.markdown("**Regression: net rating on role counts** (reference role: creator, so each coefficient is the change from swapping one creator for that role; possession-weighted; team-season fixed effects; 95% CI)")
    st.dataframe(lm.round(3), width="stretch", hide_index=True)
    st.caption("No sign of diminishing returns (squared shooter term not significant). Role mix is confounded with player quality, "
               "and 4-5 shooter lineups are very rare, so treat this as descriptive.")

# ------------------------------------------------------------------ Calculator
with tabs[7]:
    st.subheader("Break-even calculator")
    st.markdown("Is a shooter's three worth more than an average two-point attempt that season? "
                "This compares expected points per attempt only. It says nothing about win probability.")
    c = st.columns(3)
    season = c[0].selectbox("Season", SEASON_LIST, index=len(SEASON_LIST) - 1)
    pct = c[1].slider("Shooter 3P%", 0.20, 0.50, 0.36, 0.005, format="%.3f")
    zone = c[2].selectbox("Zone context", ["All threes", "Corner", "Above the break"])
    row = seasons_df[seasons_df.SEASON == season].iloc[0]
    ppa = 3 * pct
    st.metric("Expected points per three", f"{ppa:.3f}", delta=f"{ppa - row.PTS_PER_2PA:+.3f} vs average two ({row.PTS_PER_2PA:.3f})")
    st.write(f"Break-even 3P% in {season}: **{row.BREAKEVEN_3P:.1%}**. League 3P%: {row.FG3_PCT:.1%}.")
    if zone != "All threes":
        z = zones[(zones.SEASON == season) & (zones.ZONE == zone)].iloc[0]
        st.write(f"League 3P% from {zone.lower()} in {season}: **{z.FG3_PCT:.1%}** ({z.FG3A:,} attempts).")
    st.caption("Ignores free throws, offensive rebounds, shot difficulty and spacing effects.")

# ------------------------------------------------------------------ About
with tabs[8]:
    st.subheader("About this project")
    st.markdown("""
**Data:** NBA.com stats API via `nba_api` - team game logs and shot charts (2014-15 to 2025-26, regular season and playoffs),
2025-26 play-by-play, and regular-season player and five-man lineup stats.

**Methods:** logistic and OLS models with team-season fixed effects and game-clustered errors (Q1); shot-level break-even
analysis (Q2); within-shooter regressions with pre-specified cold/hot definitions and a shuffle test (cold streaks);
k-means roles plus weighted lineup regressions (Q3). Full notes in `writeup.md` and the notebooks.

**Main limits:** observational data; no defender or openness information; one season of play-by-play; shooter roles are not
cleanly separable; average in-game margin is partly an outcome, so margin-controlled models are robustness checks.

**Source:** see the README for how to rebuild the data and rerun the notebooks.
""")
