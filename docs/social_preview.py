"""Social preview image (1280x640) from an actual 1991-2026 run: python docs/social_preview.py"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from varbt.__main__ import COLOR, INK, MUTED, GRID, MODELS
from varbt.core import backtest, forecast
from varbt.data import LONG, load_yahoo

r = load_yahoo(*LONG)
passed = {}
for m in MODELS:
    b = backtest(r.values, forecast(r.values, model=m))
    passed[m] = sum([b["kupiec_p"] > 0.05, b["christoffersen_p"] > 0.05,
                     not b["tl_worst"].endswith("red"), b["z2_p"] > 0.05])
    print(MODELS[m], passed[m], "/ 4")

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
years = f"{r.index[500].year}–{r.index[-1].year}"  # first forecast day: 500 returns of history
fig.text(0.05, 0.85, "Which VaR model survives 2008 and 2020?", fontsize=34, color=INK, weight="bold")
fig.text(0.05, 0.765, f"S&P 500, {years}: 1-day 99% VaR and 97.5% Expected Shortfall, forecast daily", fontsize=17, color=MUTED)
fig.text(0.05, 0.715, "Backtests passed out of 4: Kupiec, Christoffersen, Basel traffic light, ES Z2", fontsize=17, color=MUTED)
ax = fig.add_axes([0.33, 0.14, 0.58, 0.5])
names = [MODELS[m] for m in MODELS][::-1]
vals = [passed[m] for m in MODELS][::-1]
ax.barh(names, vals, color=[COLOR[m] for m in MODELS][::-1], height=0.6)
for y, v in enumerate(vals):
    ax.text(v + 0.05, y, f"{v} / 4", va="center", fontsize=20, color=INK, weight="bold")
ax.set_xlim(0, 4.6)
ax.set_xticks([])
ax.tick_params(axis="y", labelsize=17, colors=INK, length=0)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.spines["left"].set_color(GRID)
fig.text(0.05, 0.05, "github.com/aidaman01/var-backtest", fontsize=17, color=MUTED)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_preview.png")
fig.savefig(out, dpi=100, facecolor="white")
print("->", out)
