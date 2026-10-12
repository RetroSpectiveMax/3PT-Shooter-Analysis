"""Shared logic for the Buckets-style shot charts (court, smoothing, zones, distance profiles, Plotly figures).

Grid layout (see build_shot_chart_data.py): 1-ft cells, array shape (NY, NX) = (52, 50),
column cx = floor(x_ft) + 25, row cy = floor(y_ft) + 5, basket at x = 0, y = 0 (feet).

The algorithms here are mirrored in docs/shot_explorer/app.js. Keep constants in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NX, NY, X0, Y0 = 50, 52, 25, 5
SMOOTH_SIGMA = 2.0       # court smoothing, in 1-ft cells
SHRINK_K = 3.0           # prior strength (smoothed attempts) pulling FG% toward the league
DIST_MAX = 30
THREE_PT = 23.75
COLOR_BREAKS = [-0.09, -0.03, 0.03, 0.09]   # FG% minus league average
COLORS = ["#4659a7", "#7fa0d0", "#f1e3a8", "#f4a45b", "#d1344a"]
COLOR_LABELS = ["-9%", "-3%", "+3%", "+9%"]
LEFT_COLOR, RIGHT_COLOR = "#2ca02c", "#9467bd"
PLAYER_COLOR, LEAGUE_COLOR, GROUP_COLOR = "#c8102e", "#17a2b8", "#777777"

_xs = np.arange(NX) - X0 + 0.5
_ys = np.arange(NY) - Y0 + 0.5
CELL_X, CELL_Y = np.meshgrid(_xs, _ys)           # (NY, NX) cell-center coordinates, feet
CELL_DIST = np.hypot(CELL_X, CELL_Y)


# ------------------------------------------------------------------ grids
def to_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sum rows with columns cx, cy, fga, fgm into (attempts, makes) arrays of shape (NY, NX)."""
    att = np.zeros((NY, NX))
    made = np.zeros((NY, NX))
    if len(df):
        np.add.at(att, (df.cy.to_numpy(int), df.cx.to_numpy(int)), df.fga.to_numpy(float))
        np.add.at(made, (df.cy.to_numpy(int), df.cx.to_numpy(int)), df.fgm.to_numpy(float))
    return att, made


def filter_cells(cells: pd.DataFrame, home: str = "All", result: str = "All", opp: int | None = None) -> pd.DataFrame:
    """cells: rows for one player-season. home in All/Home/Away, result in All/Wins/Losses, opp = team code."""
    if home == "Home":
        cells = cells[cells.home == 1]
    elif home == "Away":
        cells = cells[cells.home == 0]
    if result == "Wins":
        cells = cells[cells.win == 1]
    elif result == "Losses":
        cells = cells[cells.win == 0]
    if opp is not None:
        cells = cells[cells.opp == opp]
    return cells


def gaussian_kernel(sigma: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def smooth(grid: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Separable Gaussian blur with zero padding (so counts keep their meaning near the edges)."""
    k = gaussian_kernel(sigma)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, grid)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, out)


def league_fg_grid(lg_att: np.ndarray, lg_made: np.ndarray) -> np.ndarray:
    """Smoothed league FG% per cell, falling back to the overall league FG% where there are no shots."""
    overall = lg_made.sum() / max(lg_att.sum(), 1)
    sa, sm = smooth(lg_att), smooth(lg_made)
    return np.where(sa > 1e-6, (sm + overall * 1.0) / (sa + 1.0), overall)


# ------------------------------------------------------------------ marks
def color_for(diff: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(COLOR_BREAKS, diff, side="right")
    return np.array(COLORS)[idx]


def court_marks(att, made, lg_att, lg_made, mode: str = "Smoothed", min_shots: float = 2, max_shots: float = 1e9,
                fg_lo: float = 0.0, fg_hi: float = 1.0) -> pd.DataFrame:
    """Marks for the court map. Columns: x, y, att, fg, diff, size (0..1 relative frequency), color.

    Raw: every 1-ft cell with attempts. Smoothed: Gaussian-smoothed counts sampled on a 2-ft lattice with
    FG% shrunk toward the league. min_shots/max_shots filter on attempts per mark (smoothed counts for Smoothed).
    """
    lg_fg = league_fg_grid(lg_att, lg_made)
    if mode == "Raw":
        a, m = att, made
        fg = np.divide(m, a, out=np.zeros_like(a), where=a > 0)
        keep = a >= 1
    else:
        # smoothed density x 4 = shots in the 2x2 ft square around each lattice point
        a, m = smooth(att) * 4, smooth(made) * 4
        fg = (m + SHRINK_K * 4 * lg_fg) / (a + SHRINK_K * 4)
        keep = np.zeros_like(a, dtype=bool)
        keep[::2, ::2] = a[::2, ::2] >= 0.25
    keep &= (a >= min_shots) & (a <= max_shots)
    keep &= (fg >= fg_lo) & (fg <= fg_hi)
    keep &= CELL_Y <= 42
    if not keep.any():
        return pd.DataFrame(columns=["x", "y", "att", "fg", "diff", "size", "color"])
    peak = a[CELL_Y <= 42].max()
    diff = fg - lg_fg
    return pd.DataFrame({
        "x": CELL_X[keep], "y": CELL_Y[keep], "att": a[keep], "fg": fg[keep], "diff": diff[keep],
        "size": np.sqrt(a[keep] / peak), "color": color_for(diff[keep]),
    })


# zones: (name, test on cell-center x, y, dist)
def _zone_defs():
    def f(name, fn):
        return name, fn
    return [
        f("Left corner 3", lambda x, y, d: (x <= -22) & (y <= 8.94)),
        f("Right corner 3", lambda x, y, d: (x >= 22) & (y <= 8.94)),
        f("Restricted area", lambda x, y, d: d < 4),
        f("Paint (non-RA)", lambda x, y, d: (d >= 4) & (np.abs(x) <= 8) & (y <= 13.75) & (d < THREE_PT)),
        f("Mid-range left", lambda x, y, d: (d < THREE_PT) & (x < -8) & (y < 22)),
        f("Mid-range right", lambda x, y, d: (d < THREE_PT) & (x > 8) & (y < 22)),
        f("Mid-range top", lambda x, y, d: d < THREE_PT),   # whatever mid-range remains: elbows, top of key
        f("Above break 3 left", lambda x, y, d: (d >= THREE_PT) & ~((np.abs(x) >= 22) & (y <= 8.94)) & (x < -8)),
        f("Above break 3 center", lambda x, y, d: (d >= THREE_PT) & ~((np.abs(x) >= 22) & (y <= 8.94)) & (np.abs(x) <= 8)),
        f("Above break 3 right", lambda x, y, d: (d >= THREE_PT) & ~((np.abs(x) >= 22) & (y <= 8.94)) & (x > 8)),
    ]


ZONES = _zone_defs()


def zone_masks() -> list[tuple[str, np.ndarray]]:
    """Disjoint zone masks; later zones never steal cells from earlier ones."""
    taken = np.zeros((NY, NX), dtype=bool)
    out = []
    for name, fn in ZONES:
        m = fn(CELL_X, CELL_Y, CELL_DIST) & ~taken & (CELL_Y <= 42)
        taken |= m
        out.append((name, m))
    return out


def zone_table(att, made, lg_att, lg_made) -> pd.DataFrame:
    rows = []
    total = att[CELL_Y <= 42].sum()
    for name, m in zone_masks():
        a, k = att[m].sum(), made[m].sum()
        la, lk = lg_att[m].sum(), lg_made[m].sum()
        fg = k / a if a else np.nan
        lfg = lk / la if la else np.nan
        rows.append({
            "zone": name, "att": a, "fgm": k, "fg": fg, "league_fg": lfg, "diff": fg - lfg if a else np.nan,
            "freq": a / total if total else 0.0,
            "cx": float(np.average(CELL_X[m], weights=lg_att[m] + 1e-9)),
            "cy": float(np.average(CELL_Y[m], weights=lg_att[m] + 1e-9)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ distance profiles
def distance_profile(att, made) -> dict[str, np.ndarray]:
    """Per 1-ft distance bin 0..30 (cell-center distance), split by side of the basket (x < 0 left)."""
    bins = np.clip(np.floor(CELL_DIST).astype(int), 0, DIST_MAX + 1)
    valid = CELL_Y <= 42
    out = {}
    for side, mask in (("left", CELL_X < 0), ("right", CELL_X >= 0)):
        sel = mask & valid
        out[f"att_{side}"] = np.bincount(bins[sel], weights=att[sel], minlength=DIST_MAX + 2)[:DIST_MAX + 1]
        out[f"made_{side}"] = np.bincount(bins[sel], weights=made[sel], minlength=DIST_MAX + 2)[:DIST_MAX + 1]
    out["att"] = out["att_left"] + out["att_right"]
    out["made"] = out["made_left"] + out["made_right"]
    return out


def smooth_1d(v: np.ndarray, sigma: float) -> np.ndarray:
    return np.convolve(v, gaussian_kernel(sigma), mode="same")


def freq_curve(att: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    tot = att.sum()
    return smooth_1d(att, sigma) / tot * 100 if tot else np.zeros_like(att)


def fg_curve(att: np.ndarray, made: np.ndarray, prior: np.ndarray | None = None, sigma: float = 1.0) -> np.ndarray:
    """Smoothed FG% (0-100). Optional prior (FG% curve) shrinks sparse bins toward it."""
    a, m = smooth_1d(att, sigma), smooth_1d(made, sigma)
    if prior is None:
        return np.divide(m, a, out=np.full_like(a, np.nan), where=a > 0) * 100
    return (m + 3 * prior / 100) / (a + 3) * 100


# ------------------------------------------------------------------ court drawing
def court_shapes(line="#555", width=1.5) -> list[dict]:
    ln = dict(color=line, width=width)
    s = []

    def rect(x0, y0, x1, y1):
        s.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=ln))

    def arc(cx, cy, r, a0, a1, n=60):
        t = np.radians(np.linspace(a0, a1, n))
        path = "M " + " L ".join(f"{cx + r * np.cos(a):.3f},{cy + r * np.sin(a):.3f}" for a in t)
        s.append(dict(type="path", path=path, line=ln))

    rect(-25, -5.25, 25, 47)                           # outline
    rect(-8, -5.25, 8, 13.75)                          # paint
    rect(-6, -5.25, 6, 13.75)                          # lane inner
    arc(0, 13.75, 6, 0, 180)                           # free-throw circle top
    arc(0, 0, 4, 0, 180)                               # restricted area
    arc(0, 0, THREE_PT, np.degrees(np.arcsin(8.94 / THREE_PT)), 180 - np.degrees(np.arcsin(8.94 / THREE_PT)), 120)
    s.append(dict(type="line", x0=-22, y0=-5.25, x1=-22, y1=8.94, line=ln))
    s.append(dict(type="line", x0=22, y0=-5.25, x1=22, y1=8.94, line=ln))
    s.append(dict(type="line", x0=-3, y0=-1.25, x1=3, y1=-1.25, line=ln))      # backboard
    s.append(dict(type="circle", x0=-0.75, y0=-0.75, x1=0.75, y1=0.75, line=ln))
    arc(0, 41.75, 6, 180, 360)                         # center circle (visible part)
    return s


# ------------------------------------------------------------------ figures
def court_figure(marks: pd.DataFrame, mode: str, zones: pd.DataFrame | None = None, encode_freq: bool = True,
                 show_legend: bool = True, title: str = "", height: int = 520) -> go.Figure:
    fig = go.Figure()
    if mode == "Zones" and zones is not None:
        for r in zones.itertuples():
            if r.att <= 0:
                continue
            col = color_for(np.array([r.diff]))[0]
            size = 18 + (46 * np.sqrt(r.freq / max(zones.freq.max(), 1e-9)) if encode_freq else 20)
            fig.add_scatter(x=[r.cx], y=[r.cy], mode="markers+text", marker=dict(size=size, color=col, symbol="square",
                            line=dict(color="white", width=1)),
                            text=[f"{r.fg:.0%}"], textfont=dict(size=11, color="black"),
                            hovertemplate=f"{r.zone}<br>{int(r.fgm)}/{int(r.att)} FG ({r.fg:.1%})<br>"
                                          f"League {r.league_fg:.1%}<br>Share of shots {r.freq:.1%}<extra></extra>",
                            showlegend=False)
    elif len(marks):
        if mode == "Smoothed":
            sz = (6 + 16 * marks["size"]) if encode_freq else 15
        else:
            sz = (3 + 9 * marks["size"]) if encode_freq else 8
        fig.add_scatter(x=marks.x, y=marks.y, mode="markers",
                        marker=dict(size=sz, color=marks.color, symbol="square", line=dict(width=0)),
                        customdata=np.c_[marks.att, marks.fg * 100, marks["diff"] * 100],
                        hovertemplate="%{customdata[0]:.1f} shots<br>FG %{customdata[1]:.1f}%<br>"
                                      "vs league %{customdata[2]:+.1f}%<extra></extra>", showlegend=False)
    if show_legend:
        labels = ["< -9%", "-9 to -3%", "-3 to +3%", "+3 to +9%", "> +9%"]
        for c, lab in zip(COLORS, labels):
            fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=c, symbol="square"),
                            name=lab, showlegend=True)
    fig.update_layout(
        shapes=court_shapes(), height=height, title=dict(text=title, x=0.02), margin=dict(l=5, r=5, t=40, b=5),
        xaxis=dict(range=[-25.5, 25.5], visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(range=[42, -6][::-1], visible=False),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.02, x=0, title=dict(text="FG% vs league avg: ")),
    )
    # basket at the bottom, like a half-court diagram from above
    return fig


def _line(fig, x, y, name, color, row=None, col=None, dash=None, width=2):
    kw = dict(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash), legendgroup=name)
    if row:
        fig.add_scatter(row=row, col=col, **kw)
    else:
        fig.add_scatter(**kw)


def _three_line(fig):
    fig.add_vline(x=THREE_PT, line=dict(color="#999", dash="dot", width=1), annotation_text="3pt",
                  annotation_position="top right")


def _colors(profiles: dict) -> list[str]:
    """First entry red, 'League' teal, a second player blue, a position group grey."""
    out, n = [], 0
    for label in profiles:
        if label == "League":
            out.append(LEAGUE_COLOR)
        elif n == 0:
            out.append(PLAYER_COLOR); n += 1
        elif label in ("Guards", "Forwards", "Bigs"):
            out.append(GROUP_COLOR)
        else:
            out.append("#1f4e9c"); n += 1
    return out


def frequency_figure(profiles: dict[str, dict], height: int = 330) -> go.Figure:
    """profiles: label -> distance_profile dict. First entry is the player (red)."""
    fig = go.Figure()
    d = np.arange(DIST_MAX + 1)
    for (label, p), color in zip(profiles.items(), _colors(profiles)):
        _line(fig, d, freq_curve(p["att"]), label, color)
    _three_line(fig)
    fig.update_layout(title="Shot frequency % by distance", height=height, xaxis_title="Distance (ft)",
                      yaxis_title="Frequency %", margin=dict(l=10, r=10, t=60, b=10), title_x=0.01, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig


def fg_figure(profiles: dict[str, dict], height: int = 330) -> go.Figure:
    fig = go.Figure()
    d = np.arange(DIST_MAX + 1)
    league = None
    for label, p in profiles.items():
        if label == "League":
            league = fg_curve(p["att"], p["made"])
    for (label, p), color in zip(profiles.items(), _colors(profiles)):
        prior = league if label != "League" else None
        _line(fig, d, fg_curve(p["att"], p["made"], prior), label, color)
    _three_line(fig)
    fig.update_layout(title="Field goal % by distance", height=height, xaxis_title="Distance (ft)",
                      yaxis_title="Field goal %", yaxis_range=[0, 100], margin=dict(l=10, r=10, t=60, b=10), title_x=0.01,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig


def left_right_figure(p: dict[str, np.ndarray], kind: str = "freq", height: int = 420) -> go.Figure:
    """Mirrored left/right bars by distance. kind: 'freq' (# of shots) or 'fg' (field goal %)."""
    d = np.arange(DIST_MAX + 1)
    if kind == "freq":
        left, right = p["att_left"], p["att_right"]
        lsm, rsm = smooth_1d(left, 0.8), smooth_1d(right, 0.8)
        title, xt = "Shot frequency: left side vs. right side", "# of shots"
        diff = rsm - lsm
        lv, rv = -left, right
        lo, hi = lsm.sum() / max(lsm.sum() + rsm.sum(), 1e-9), rsm.sum() / max(lsm.sum() + rsm.sum(), 1e-9)
    else:
        lsmooth_a, lsmooth_m = smooth_1d(p["att_left"], 1.0), smooth_1d(p["made_left"], 1.0)
        rsmooth_a, rsmooth_m = smooth_1d(p["att_right"], 1.0), smooth_1d(p["made_right"], 1.0)
        lfg = np.divide(lsmooth_m, lsmooth_a, out=np.zeros_like(lsmooth_a), where=lsmooth_a > 0) * 100
        rfg = np.divide(rsmooth_m, rsmooth_a, out=np.zeros_like(rsmooth_a), where=rsmooth_a > 0) * 100
        title, xt = "Field goal %: left side vs. right side", "Field goal %"
        lv, rv = -lfg, rfg
        diff = rfg - lfg
        la, ra = p["att_left"].sum(), p["att_right"].sum()
        lo = p["made_left"].sum() / la if la else 0
        hi = p["made_right"].sum() / ra if ra else 0
    fig = go.Figure()
    fig.add_bar(y=d, x=lv, orientation="h", name="Left", marker_color=LEFT_COLOR, opacity=0.35, width=1)
    fig.add_bar(y=d, x=rv, orientation="h", name="Right", marker_color=RIGHT_COLOR, opacity=0.35, width=1)
    fig.add_scatter(y=d, x=diff, mode="lines", name="Difference (R - L)", line=dict(color="#333", width=2))
    fig.add_hline(y=THREE_PT, line=dict(color="#999", dash="dot", width=1))
    fig.update_layout(barmode="overlay", title=f"{title}<br><sup>Overall: left {lo:.0%} | right {hi:.0%}</sup>",
                      height=height, xaxis_title=xt, yaxis_title="Distance (ft)", yaxis_range=[0, DIST_MAX],
                      margin=dict(l=10, r=10, t=90, b=10), title_x=0.01, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    if kind == "fg":
        fig.update_xaxes(range=[-100, 100], tickvals=[-100, -50, 0, 50, 100], ticktext=["100%", "50%", "0%", "50%", "100%"])
    else:
        fig.update_xaxes(tickformat="s")
    return fig
