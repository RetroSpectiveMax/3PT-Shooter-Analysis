"""Make the README figures from the summary tables in data/app/.

Usage: python src/make_figures.py   (writes docs/figures/*.png)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "data" / "app"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def rd(n):
    return pd.read_parquet(APP / f"{n}.parquet")


def breakeven():
    s = rd("seasons")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s.SEASON, s.FG3_PCT * 100, "o-", label="League 3P%")
    ax.plot(s.SEASON, s.BREAKEVEN_3P * 100, "s--", label="Break-even 3P%")
    ax.fill_between(s.SEASON, s.BREAKEVEN_3P * 100, s.FG3_PCT * 100, alpha=0.12)
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title("The average three has caught up to the average two")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "breakeven_vs_league.png")


def curve():
    c = rd("curves")
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, d in c.groupby("MODEL"):
        ax.plot(d.FG3A, d.EFFECT, label=m)
        ax.fill_between(d.FG3A, d.LO, d.HI, alpha=0.12)
    ax.axhline(0, c="gray", ls="--")
    ax.set_xlabel("team 3PA per game")
    ax.set_ylabel("predicted margin vs 30 3PA (points)")
    ax.set_title("No clear optimum: adjusted curves are flat or convex, never a hill")
    ax.legend(fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "adjusted_3pa_curve.png")


def zones():
    z = rd("zones")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, d in z.groupby("ZONE"):
        ax.plot(d.SEASON, d.FG3_PCT * 100, "o-", label=name)
    ax.set_ylabel("3P%")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title("Corner threes are about 3 points better than above-the-break")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "corner_vs_above_break.png")


def cold():
    g = rd("cold_grid")
    g = g[(((g.GROUP == "cold") & (g.CUTOFF == 0.25)) | ((g.GROUP == "hot") & (g.CUTOFF == 0.50))) & g.N.isin([8, 10, 15])]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (grp, col) in enumerate([("cold", "tab:blue"), ("hot", "tab:red")]):
        d = g[g.GROUP == grp].sort_values("N")
        ax.errorbar(d.N + (i - 0.5) * 0.5, d.EFFECT_PP, yerr=[d.EFFECT_PP - d.LO, d.HI - d.EFFECT_PP],
                    fmt="o", color=col, capsize=3, label=grp)
    ax.axhline(0, c="gray", ls="--")
    ax.set_xticks([8, 10, 15])
    ax.set_xlabel("window N (previous team threes in the game)")
    ax.set_ylabel("effect on next-three make probability (pp)")
    ax.set_title("Cold streaks do not predict the next three (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cold_streak.png")


def state():
    s = rd("score_state").sort_values("ORDER")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s.STATE, s.SHARE_3 * 100, "o-")
    ax.set_ylim(30, 50)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("threes as % of shots")
    ax.set_title("Trailing teams do not shoot more threes (2025-26)")
    fig.tight_layout()
    fig.savefig(OUT / "three_share_by_score_state.png")


if __name__ == "__main__":
    for f in (breakeven, curve, zones, cold, state):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
