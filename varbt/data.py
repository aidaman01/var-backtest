"""Daily index returns: Yahoo Finance via yfinance (downloaded once, cached in data/, never committed),
any date,price CSV, or simulated GARCH-t (known truth)."""
import os

import numpy as np
import pandas as pd

# The two published runs: (ticker, start, end); yfinance's end date is exclusive.
MAIN = ("^GSPC", "2016-02-12", "2026-02-12")   # last day 2026-02-11
LONG = ("^GSPC", "1990-01-01", "2026-10-15")   # last day 2026-10-14


def load_csv(path):
    """CSV with dates in the first column and prices in the second (blank prices skipped).
    Returns daily log returns indexed by date."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    px = pd.to_numeric(px, errors="coerce").dropna()
    return np.log(px).diff().dropna()


def load_yahoo(ticker="^GSPC", start=MAIN[1], end=MAIN[2], cache="data"):
    """Daily closes from Yahoo Finance, cached as data/<ticker>_<start>_<end>.csv. Yahoo's terms allow
    personal use only, so the data is downloaded on each machine instead of shipped with the code."""
    path = os.path.join(cache, f"{ticker.lstrip('^')}_{start}_{end}.csv")
    if not os.path.exists(path):
        import yfinance as yf

        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
        if len(px) < 1000:
            raise RuntimeError(f"only {len(px)} prices for {ticker}; check the ticker or connection")
        os.makedirs(cache, exist_ok=True)
        px.rename("close").to_csv(path, index_label="date")
    return load_csv(path)


def simulate_garch_t(n, mu=0.0003, omega=2e-6, a=0.10, b=0.88, nu=5.0, seed=0, g=0.0):
    """r = mu + sigma z, sigma2 GARCH(1,1) (GJR if g > 0), z unit-variance Student-t. Returns (r, sigma)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    s2, r = np.empty(n), np.empty(n)
    s2[0] = omega / (1 - a - g / 2 - b)
    for t in range(n):
        if t:
            e = r[t - 1] - mu
            s2[t] = omega + (a + g * (e < 0)) * e ** 2 + b * s2[t - 1]
        r[t] = mu + np.sqrt(s2[t]) * z[t]
    return r, np.sqrt(s2)
