"""Daily index returns: open CSV (FRED S&P 500), live via yfinance, or simulated GARCH-t (known truth)."""
import numpy as np
import pandas as pd


def load_csv(path):
    """FRED-style CSV (observation_date, SP500). Returns daily log returns indexed by date."""
    d = pd.read_csv(path, parse_dates=["observation_date"]).dropna()
    px = d.set_index("observation_date")["SP500"].astype(float)
    return np.log(px).diff().dropna()


def load_yahoo(ticker="^GSPC", start="1990-01-01"):
    import yfinance as yf

    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    if len(px) < 1000:
        raise RuntimeError(f"only {len(px)} prices for {ticker}; check the ticker or connection")
    return np.log(px).diff().dropna()


def simulate_garch_t(n, mu=0.0003, omega=2e-6, a=0.10, b=0.88, nu=5.0, seed=0):
    """r = mu + sigma z, sigma2 GARCH(1,1), z unit-variance Student-t. Returns (r, sigma)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    s2, r = np.empty(n), np.empty(n)
    s2[0] = omega / (1 - a - b)
    for t in range(n):
        if t:
            s2[t] = omega + a * (r[t - 1] - mu) ** 2 + b * s2[t - 1]
        r[t] = mu + np.sqrt(s2[t]) * z[t]
    return r, np.sqrt(s2)
