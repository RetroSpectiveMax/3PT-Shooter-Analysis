"""Generate the analysis notebooks from cell lists (then execute with nbconvert).

Usage: python src/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
        for c in cells
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)


Q1 = [
("md", """# Q1: What is the optimal number of team 3-point attempts per game?

Unit: one row per team per game, 2014-15 to 2025-26. Naive comparisons are
confounded (good teams shoot more threes partly because they are good), so
the models add team-season fixed effects and opponent strength.

**Limit:** the box score has no in-game score margin, so we cannot control for
trailing teams shooting more threes here. Revisit with play-by-play data."""),
("code", """import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from src.features import load_team_games, vertex_ci
plt.rcParams['figure.dpi'] = 110
g = load_team_games()
print(len(g), 'team-games')
g.groupby('SEASON')['FG3A'].mean().round(1).to_frame('mean 3PA per team-game').T"""),
("md", "## 1. Raw relationship: win rate and margin by 3PA bucket"),
("code", """bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 100]
g['BUCKET'] = pd.cut(g['FG3A'], bins=bins)
t = g.groupby('BUCKET', observed=True).agg(games=('WIN','size'), win_rate=('WIN','mean'), margin=('MARGIN','mean'))
t['se'] = np.sqrt(t.win_rate*(1-t.win_rate)/t.games)
display(t.round(3))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = range(len(t))
ax[0].errorbar(x, t.win_rate, yerr=1.96*t.se, fmt='o-'); ax[0].axhline(.5, c='gray', ls='--')
ax[0].set_xticks(x); ax[0].set_xticklabels([str(i) for i in t.index], rotation=45)
ax[0].set_title('Win rate by 3PA bucket (raw)'); ax[0].set_ylabel('win rate')
ax[1].bar(x, t.margin); ax[1].set_xticks(x); ax[1].set_xticklabels([str(i) for i in t.index], rotation=45)
ax[1].set_title('Mean point margin by 3PA bucket (raw)')
plt.tight_layout(); plt.show()"""),
("md", """## 2. Era check: the raw curve by period

League volume rose from ~22 to ~37 per team-game, so 'many threes' means different things in different years."""),
("code", """g['ERA'] = pd.cut(g['SEASON'].str[:4].astype(int), [2013, 2017, 2021, 2025], labels=['2014-18','2018-22','2022-26'])
for era, d in g.groupby('ERA', observed=True):
    r = d.groupby(pd.cut(d.FG3A, bins), observed=True)['WIN'].agg(['mean','size'])
    r = r[r['size'] >= 100]
    plt.plot([str(i) for i in r.index], r['mean'], 'o-', label=str(era))
plt.axhline(.5, c='gray', ls='--'); plt.xticks(rotation=45); plt.legend(); plt.ylabel('win rate')
plt.title('Win rate by 3PA bucket and era (buckets with 100+ games)'); plt.show()"""),
("md", """## 3. Models

All use standard errors clustered by game (each game appears twice).

- **Total effect:** `WIN ~ FG3A + FG3A^2` with team-season fixed effects, opponent strength and home. Everything that moves with 3PA (accuracy, shot mix) is allowed to flow through.
- **Full box-score model:** adds 3P%, 2PA, FTA, TOV, OREB, pace. This holds those constant, so it asks what an *extra attempt at fixed accuracy* does. Note 3P% in the same game is partly luck, which is an outcome not a choice."""),
("code", """def fit(formula, kind='logit'):
    f = smf.logit if kind == 'logit' else smf.ols
    return f(formula, data=g).fit(disp=0, cov_type='cluster', cov_kwds={'groups': g['GAME_ID']})

fe = 'C(TEAM_SEASON) + OPP_STRENGTH + HOME'
models = {
 'logit_total': fit(f'WIN ~ FG3A + FG3A_SQ + {fe}'),
 'logit_full' : fit(f'WIN ~ FG3A + FG3A_SQ + FG3_PCT + FG2A + FTA + TOV + OREB + POSS + {fe}'),
 'ols_total'  : fit(f'MARGIN ~ FG3A + FG3A_SQ + {fe}', 'ols'),
 'ols_full'   : fit(f'MARGIN ~ FG3A + FG3A_SQ + FG3_PCT + FG2A + FTA + TOV + OREB + POSS + {fe}', 'ols'),
}
rows = []
for k, m in models.items():
    p, se = m.params, m.bse
    est, lo, hi = vertex_ci(p, m.cov_params(), 'FG3A', 'FG3A_SQ')
    rows.append(dict(model=k, b_FG3A=p['FG3A'], se=se['FG3A'], b_FG3A_SQ=p['FG3A_SQ'], se_sq=se['FG3A_SQ'],
                     p_sq=m.pvalues['FG3A_SQ'], vertex=est, ci_lo=lo, ci_hi=hi))
res = pd.DataFrame(rows).set_index('model'); res.round(4)"""),
("md", """A vertex only means 'the optimum' if the squared term is negative and significant. If it is not, the data do not show diminishing returns in this range, and the vertex CI will be very wide or unbounded."""),
("code", """# Predicted effect of 3PA on win probability, holding other things at their mean (total-effect model)
m = models['logit_total']
base = g.sample(5000, random_state=0).copy()
xs = np.arange(15, 56)
curve = []
for x in xs:
    d = base.copy(); d['FG3A'] = x; d['FG3A_SQ'] = x**2
    curve.append(m.predict(d).mean())
curve = np.array(curve)
plt.plot(xs, curve); plt.axhline(.5, c='gray', ls='--')
plt.hist(g.FG3A, bins=40, density=True, alpha=.2, color='k', label='share of team-games (scaled)', bottom=curve.min())
plt.xlabel('team 3PA'); plt.ylabel('predicted win prob (avg team/opp)')
plt.title('Adjusted win probability vs 3PA (team-season FE + opponent strength)'); plt.show()
pd.Series(curve, index=xs).iloc[::5].round(3)"""),
("md", "## 4. Does the optimum shift by era? (total-effect OLS on margin, by period)"),
("code", """out = []
for era, d in g.groupby('ERA', observed=True):
    mm = smf.ols(f'MARGIN ~ FG3A + FG3A_SQ + {fe}', data=d).fit(cov_type='cluster', cov_kwds={'groups': d['GAME_ID']})
    est, lo, hi = vertex_ci(mm.params, mm.cov_params(), 'FG3A', 'FG3A_SQ')
    out.append(dict(era=era, b_sq=mm.params['FG3A_SQ'], p_sq=mm.pvalues['FG3A_SQ'], mean_3PA=d.FG3A.mean(), vertex=est, lo=lo, hi=hi))
pd.DataFrame(out).set_index('era').round(3)"""),
("md", "## 5. Gradient boosting robustness check (no assumed curve)"),
("code", """from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
feats = ['FG3A','FG3_PCT','FG2A','FTA','TOV','OREB','POSS','HOME','OPP_STRENGTH','TEAM_STRENGTH']
d = g.dropna(subset=feats+['MARGIN']).copy()
d[feats] = d[feats].astype(float)
# team-strength is leave-one-game-out, so it is a fair control
gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=300, random_state=0).fit(d[feats], d['MARGIN'])
pdp = partial_dependence(gb, d[feats], ['FG3A'], grid_resolution=40)
plt.plot(pdp['grid_values'][0], pdp['average'][0]); plt.xlabel('team 3PA'); plt.ylabel('predicted margin')
plt.title('Partial dependence of margin on 3PA (boosting)'); plt.show()"""),
("md", "## 6. High-volume extremes (sanity check)"),
("code", """s = g.groupby(['TEAM_NAME','SEASON']).agg(games=('WIN','size'), FG3A=('FG3A','mean'), FG3_PCT=('FG3M', 'sum'),
                                          FG3A_tot=('FG3A','sum'), win_pct=('WIN','mean'), margin=('MARGIN','mean'))
s['FG3_PCT'] = s['FG3_PCT']/s['FG3A_tot']
s.drop(columns='FG3A_tot').sort_values('FG3A', ascending=False).head(8).round(3)"""),
("md", """## Reading the results

See the tables above. Key cautions:

- Team-season fixed effects remove the 'good teams shoot more' problem between teams but not within a team over games (a team shoots more when trailing).
- In-game score state is not in this data. Play-by-play for 2025-26 allows a follow-up.
- Estimates describe the observed range (about 15-55 3PA). Do not extrapolate beyond it."""),
]


Q2 = [
("md", """# Q2: What 3P% breaks even, and does it change with volume?

A three is worth 3 points, so a three beats an average two-point attempt when

    3P% > (points per 2-point attempt) / 3

Shot-level data: every FGA, 2014-15 to 2025-26 (regular season + playoffs)."""),
("code", """import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.features import load_shots, load_team_games, breakeven_3p
plt.rcParams['figure.dpi'] = 110
s = load_shots()
s['IS3'] = (s.SHOT_TYPE == '3PT Field Goal')
s['PTS'] = s.SHOT_MADE_FLAG * np.where(s.IS3, 3, 2)
print(f'{len(s):,} shots')"""),
("md", "## 1. League break-even by season"),
("code", """g = load_team_games()
ft = g.groupby('SEASON').agg(FTA=('FTA','sum'), FTM=('FTM','sum'), FGA=('FGA','sum'))
by = s.groupby('SEASON').apply(lambda d: pd.Series({
    'n2': (~d.IS3).sum(), 'ppa2': d.loc[~d.IS3, 'PTS'].mean(), 'fg2': d.loc[~d.IS3, 'SHOT_MADE_FLAG'].mean(),
    'n3': d.IS3.sum(), 'fg3': d.loc[d.IS3, 'SHOT_MADE_FLAG'].mean()}), include_groups=False)
by['breakeven_3p'] = breakeven_3p(by.ppa2)
by['edge_pts_per_3PA'] = 3*by.fg3 - by.ppa2          # points gained per three vs average two
by['share_3PA'] = by.n3/(by.n3+by.n2)
# Fuller measure: add free-throw points per FGA (all attempts, so a shot-neutral adjustment)
by['ft_pts_per_fga'] = ft.FTM/ft.FGA
display(by[['fg2','ppa2','fg3','breakeven_3p','edge_pts_per_3PA','share_3PA','ft_pts_per_fga']].round(3))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(by.index, by.fg3, 'o-', label='league 3P%'); ax[0].plot(by.index, by.breakeven_3p, 's--', label='break-even 3P%')
ax[0].legend(); ax[0].tick_params(axis='x', rotation=60); ax[0].set_title('League 3P% vs break-even')
ax[1].bar(by.index, by.edge_pts_per_3PA); ax[1].tick_params(axis='x', rotation=60); ax[1].set_title('Points per three above an average two')
plt.tight_layout(); plt.show()"""),
("md", """Free throws: a drawn foul on a two is worth extra points that a three rarely carries, so the pure shot-based break-even is the cleaner comparison; the free-throw column above is context only. Shot-level data does not tell us which attempts drew fouls (and-ones appear only as made shots)."""),
("md", "## 2. Where the threes are taken: corner vs above the break"),
("code", """z = s[s.IS3 & s.SHOT_ZONE_BASIC.isin(['Left Corner 3','Right Corner 3','Above the Break 3'])].copy()
z['zone'] = np.where(z.SHOT_ZONE_BASIC == 'Above the Break 3', 'Above the break', 'Corner')
zt = z.groupby(['SEASON','zone']).SHOT_MADE_FLAG.agg(['mean','size']).unstack()
display(zt.round(3))
zt['mean'].plot(marker='o'); plt.ylabel('3P%'); plt.title('3P% by zone'); plt.show()"""),
("md", """## 3. Does efficiency fall as shooter volume rises?

Player-season, minimum 100 3PA. Volume bucket = 3PA per team game the player's team played that season (a per-team-game rate, so players are comparable regardless of minutes)."""),
("code", """tg = g.groupby(['SEASON','TEAM_ID']).size().rename('team_games').reset_index()
p = s[s.IS3].groupby(['SEASON','PLAYER_ID','PLAYER_NAME','TEAM_ID']).SHOT_MADE_FLAG.agg(fg3a='size', fg3m='sum').reset_index()
# players traded mid-season: combine across teams, use the larger team-game count
p = p.merge(tg, on=['SEASON','TEAM_ID'])
p = p.groupby(['SEASON','PLAYER_ID','PLAYER_NAME']).agg(fg3a=('fg3a','sum'), fg3m=('fg3m','sum'), team_games=('team_games','max')).reset_index()
p = p[p.fg3a >= 100].copy()
p['fg3_pct'] = p.fg3m/p.fg3a
p['rate'] = p.fg3a/p.team_games
p = p.merge(by[['breakeven_3p','ppa2']].reset_index(), on='SEASON')
p['bucket'] = pd.cut(p.rate, [0, 2, 3, 4, 5, 6, 20], labels=['<2','2-3','3-4','4-5','5-6','6+'])
bt = p.groupby('bucket', observed=True).apply(lambda d: pd.Series({
    'players': len(d), 'fg3a_total': d.fg3a.sum(), 'fg3_pct': d.fg3m.sum()/d.fg3a.sum(),
    'mean_breakeven': d.breakeven_3p.mean(),
    'pts_per_3PA': 3*d.fg3m.sum()/d.fg3a.sum(),
    'share_above_breakeven': (d.fg3_pct > d.breakeven_3p).mean()}), include_groups=False)
bt['edge'] = bt.pts_per_3PA - 3*bt.mean_breakeven
display(bt.round(3))"""),
("code", """# Within-player check: does the same player shoot worse in seasons where he shoots more?
from statsmodels.formula.api import ols
pp = p.copy()
pp['pid'] = pp.PLAYER_ID.astype(str)
m = ols('fg3_pct ~ rate + C(pid) + C(SEASON)', data=pp[pp.groupby('PLAYER_ID').PLAYER_ID.transform('size') >= 2]).fit(
    cov_type='cluster', cov_kwds={'groups': pp[pp.groupby('PLAYER_ID').PLAYER_ID.transform('size') >= 2].PLAYER_ID})
print(f"within-player: +1 3PA per team game -> {m.params['rate']*100:+.2f} pts of 3P% (95% CI {m.conf_int().loc['rate',0]*100:+.2f} to {m.conf_int().loc['rate',1]*100:+.2f})")"""),
("md", "## 4. Shooters against the break-even line (latest season and all seasons)"),
("code", """fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for a, d, ttl in [(ax[0], p[p.SEASON == '2025-26'], '2025-26'), (ax[1], p, 'All seasons')]:
    a.scatter(d.rate, d.fg3_pct, s=d.fg3a/8, alpha=.4)
    be = d.breakeven_3p.mean(); a.axhline(be, c='r', ls='--', label=f'break-even ~ {be:.3f}')
    a.set_xlabel('3PA per team game'); a.set_title(ttl); a.legend()
ax[0].set_ylabel('3P%'); plt.tight_layout(); plt.show()"""),
("md", """## Reading the results

- The break-even 3P% is about the same as league 3P% in most seasons, so the average three is only mildly better than the average two. This is the league-wide shot-selection picture: it assumes the shooter and shot location are typical.
- Efficiency vs volume is confounded by who gets to shoot (high-volume shooters are usually the best) and by shot difficulty. The within-player check is the cleaner read.
- Shot-quality context (open vs contested) is not in this data; zone and distance are the only quality proxies used."""),
]

if __name__ == "__main__":
    NB.mkdir(exist_ok=True)
    build("01_q1_optimal_3pa.ipynb", Q1)
    build("02_q2_breakeven.ipynb", Q2)
    print("built")
