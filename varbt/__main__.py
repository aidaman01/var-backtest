"""python -m varbt [--csv FILE | --ticker ^GSPC] [--out var.png]"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import backtest, forecast, traffic_light_limits
from .data import load_csv, load_yahoo

MODELS = {"hs": "Historical simulation", "ewma": "EWMA normal (RiskMetrics)", "fhs": "Filtered HS (GARCH-t)",
          "fhs_gjr": "Filtered HS (GJR-GARCH-t)"}
COLOR = {"hs": "#eb6834", "ewma": "#1baf7a", "fhs": "#2a78d6", "fhs_gjr": "#8b5cf6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def report(r, res):
    dates = r.index
    print(f"\nS&P 500 daily returns {dates[0].date()} to {dates[-1].date()}; "
          f"backtest {dates[res['fhs'][0]['days'][0]].date()} onward ({res['fhs'][1]['days']} days)")
    print(f"{'model':28}{'99% VaR exc.':>14}{'Kupiec p':>10}{'Christ. p':>11}"
          f"{'last 250d':>13}{'worst 250d':>13}{'ES Z2':>8}{'Z2 p':>7}{'avg VaR':>9}")
    for m, (f, b) in res.items():
        print(f"{MODELS[m]:28}{b['exceptions']:>7} / {b['expected']:<5.1f}{b['kupiec_p']:>10.3f}"
              f"{b['christoffersen_p']:>11.3f}{b['tl_last']:>13}{b['tl_worst']:>13}{b['z2']:>8.2f}{b['z2_p']:>7.3f}"
              f"{f['var99'].mean() * 100:>8.2f}%")
        print(f"{'':28}worst 250-day window ends {dates[f['days'][b['tl_worst_end']]].date()}")
    print("p < 0.05 rejects the model. Traffic light: 99% VaR exceptions in 250 days, green <= 4, red >= 10.")


def plot(r, res, out):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for a in (a1, a2):
        a.grid(color=GRID, lw=0.8)
        a.spines[["top", "right"]].set_visible(False)

    days = res["fhs"][0]["days"]
    t = r.index[days]
    loss = -r.values[days] * 100
    a1.vlines(t, 0, np.clip(loss, 0, None), color="#8a8984", lw=0.7, label="daily loss")
    for m, (f, b) in res.items():
        a1.plot(t, f["var99"] * 100, color=COLOR[m], lw=1.1, label=f"99% VaR, {MODELS[m]}")
    fb = res["fhs_gjr"][1]["hits"]
    a1.plot(t[fb], loss[fb], "o", ms=4, mfc="none", mec=COLOR["fhs_gjr"], label="loss beyond FHS-GJR VaR")
    top = loss.max() * 1.25
    peak_right, n_peaks = False, 0
    for m, (f, b) in res.items():  # label any VaR peak cut off by the axis, stacked
        i = int(np.argmax(f["var99"]))
        if f["var99"][i] * 100 > top:
            a1.annotate(f"{MODELS[m]} VaR peak {f['var99'][i] * 100:.0f}%", (t[i], top * 0.97),
                        xytext=(8, -4 - 12 * n_peaks), textcoords="offset points", fontsize=8, color=COLOR[m], va="top")
            peak_right |= i > len(t) / 2
            n_peaks += 1
    a1.set(ylabel="loss (%)", title="One-day 99% Value at Risk vs realised S&P 500 losses", ylim=(0, top),
           xlim=(t[0], t[-1]))
    a1.legend(frameon=False, fontsize=9, loc="upper left" if peak_right else "upper right")

    g, y = traffic_light_limits()
    a2.axhspan(-0.5, g + 0.5, color="#e3f1e3", lw=0)
    a2.axhspan(g + 0.5, y + 0.5, color="#fbf0d4", lw=0)
    a2.axhspan(y + 0.5, 30, color="#f8e0dc", lw=0)
    for m, (f, b) in res.items():
        c = np.convolve(b["hits"].astype(int), np.ones(250, int), "full")[:len(b["hits"])]
        a2.step(t, c, where="post", color=COLOR[m], lw=1.6, label=MODELS[m])
    top = max(15, max(np.convolve(b["hits"], np.ones(250))[: len(b["hits"])].max() for _, b in res.values()) + 2)
    a2.set(ylim=(0, top), ylabel="exceptions, last 250 days", title="Basel traffic light (green / yellow / red zones)")
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"chart -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Backtest 1-day VaR and ES models on an equity index.")
    ap.add_argument("--csv", default=None, help="FRED-style CSV (observation_date, SP500)")
    ap.add_argument("--ticker", default=None, help="live via yfinance, e.g. ^GSPC")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--out", default="var.png")
    a = ap.parse_args()
    if a.ticker:
        r = load_yahoo(a.ticker, a.start)
    else:
        r = load_csv(a.csv or "tests/data/sp500_fred.csv")
    res = {}
    for m in MODELS:
        f = forecast(r.values, model=m)
        res[m] = (f, backtest(r.values, f))
    report(r, res)
    plot(r, res, a.out)


if __name__ == "__main__":
    main()
