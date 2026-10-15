"""1-day VaR and Expected Shortfall for an index, and the backtests that judge them.

Models (all use only data before the forecast day):
  hs    Historical simulation: the last `window` returns are the scenarios (Basel default).
  ewma  RiskMetrics: normal with EWMA volatility, lambda = 0.94.
  fhs   Filtered historical simulation: GARCH(1,1)-t volatility x empirical standardized residuals.

Backtests:
  Kupiec POF (exception count), Christoffersen independence and conditional coverage,
  Basel traffic light for 99% VaR, Acerbi-Szekely Z2 for 97.5% ES (FRTB) with a Monte Carlo p-value.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import binom, chi2, norm
from scipy.stats import t as student_t

VAR_LEVEL, ES_LEVEL = 0.99, 0.975


# ---------- GARCH(1,1) with Student-t innovations ----------

def garch_variance(e, omega, a, b, v0):
    """sigma2[t] = omega + a e[t-1]^2 + b sigma2[t-1]; sigma2[0] = v0. Returns len(e)+1 values
    (the last one is the forecast for the next day)."""
    u = np.concatenate([[v0], omega + a * e ** 2])
    return lfilter([1.0], [1.0, -b], u)  # y[t] = u[t] + b y[t-1], y[0] = v0


def garch_fit(r):
    """MLE of r = mu + sigma z, z ~ unit-variance Student-t. Returns (mu, omega, a, b, nu).
    Fitted on r / sd with a = pers * share, b = pers * (1 - share), so stationarity is a plain box
    bound: a penalty wall at a + b = 0.999 used to stall L-BFGS-B up to 6 log-lik units short (measured)."""
    sd = r.std()
    y = r / sd

    def nll(p):
        mu, lw, pers, share, nu = p
        e = y - mu
        s2 = garch_variance(e, np.exp(lw), pers * share, pers * (1 - share), 1.0)[:-1]
        k = np.sqrt(nu / (nu - 2))
        return -np.sum(student_t.logpdf(e / np.sqrt(s2) * k, nu) + np.log(k) - 0.5 * np.log(s2))

    x0 = [y.mean(), np.log(0.05), 0.96, 0.08, 6.0]
    bounds = [(-0.01 / sd, 0.01 / sd), (-12, 0), (0.0, 0.998), (1e-4, 1.0), (2.1, 50)]
    mu, lw, pers, share, nu = minimize(nll, x0, method="L-BFGS-B", bounds=bounds).x
    return mu * sd, np.exp(lw) * sd ** 2, pers * share, pers * (1 - share), nu


# ---------- rolling forecasts ----------

def _tail(S, level):
    """VaR and ES (positive losses) at `level` from a scenario matrix S (days x scenarios).
    method="weibull" (plotting position k/(n+1)) gives the right exception rate on average;
    numpy's default "linear" puts the 1% quantile of 500 points ~20% too far in (measured)."""
    q = np.quantile(S, 1 - level, axis=1, method="weibull")
    tail = np.where(S <= q[:, None], S, np.nan)
    return -q, -np.nanmean(tail, axis=1)


def forecast(r, window=500, model="fhs", refit=21, lam=0.94):
    """Forecasts for days window..len(r)-1. Returns dict with var99, var975, es975 and a sampler
    `draw(rng, M)` of the model's own predictive distribution (used for the ES test p-value)."""
    r = np.asarray(r, float)
    days = np.arange(window, len(r))

    if model == "ewma":
        s2 = np.empty(len(r) + 1)
        s2[0] = r[:window].var()  # initialisation only; first forecast is `window` days later
        for t in range(len(r)):
            s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
        sig = np.sqrt(s2[days])
        z99, z975 = norm.ppf(1 - VAR_LEVEL), norm.ppf(1 - ES_LEVEL)
        return dict(days=days, var99=-sig * z99, var975=-sig * z975,
                    es975=sig * norm.pdf(z975) / (1 - ES_LEVEL),
                    draw=lambda rng, M: sig * rng.standard_normal((M, len(days))))

    idx = days[:, None] + np.arange(-window, 0)  # past `window` days for each forecast day
    if model == "hs":
        S = r[idx]
    elif model == "fhs":
        S = np.empty(idx.shape)
        for i0 in range(0, len(days), refit):
            t0 = days[i0]
            mu, omega, a, b, nu = garch_fit(r[t0 - window:t0])
            s = np.sqrt(garch_variance(r - mu, omega, a, b, r[:window].var()))  # s[t]: info to t-1
            blk = slice(i0, min(i0 + refit, len(days)))
            z = (r[idx[blk]] - mu) / s[idx[blk]]
            S[blk] = mu + s[days[blk], None] * z
    else:
        raise ValueError(model)

    v99, _ = _tail(S, VAR_LEVEL)
    v975, e975 = _tail(S, ES_LEVEL)
    n = S.shape[1]
    return dict(days=days, var99=v99, var975=v975, es975=e975,
                draw=lambda rng, M: S[np.arange(len(days)), rng.integers(0, n, (M, len(days)))])


# ---------- backtests ----------

def _xlogy(x, y):
    return np.where(x > 0, x * np.log(np.where(y > 0, y, 1)), 0.0)


def kupiec(hits, p):
    """Proportion-of-failures LR test. Returns (LR, p-value)."""
    T, x = len(hits), int(np.sum(hits))
    ph = x / T
    lr = -2 * (_xlogy(T - x, 1 - p) + _xlogy(x, p) - _xlogy(T - x, 1 - ph) - _xlogy(x, ph))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen(hits):
    """Independence LR test: is an exception more likely the day after an exception? (LR, p-value)."""
    h = np.asarray(hits, int)
    prev, cur = h[:-1], h[1:]
    n = {(i, j): int(np.sum((prev == i) & (cur == j))) for i in (0, 1) for j in (0, 1)}
    p01 = n[0, 1] / max(n[0, 0] + n[0, 1], 1)
    p11 = n[1, 1] / max(n[1, 0] + n[1, 1], 1)
    p1 = (n[0, 1] + n[1, 1]) / len(cur)
    l1 = (_xlogy(n[0, 0], 1 - p01) + _xlogy(n[0, 1], p01) + _xlogy(n[1, 0], 1 - p11) + _xlogy(n[1, 1], p11))
    l0 = _xlogy(n[0, 0] + n[1, 0], 1 - p1) + _xlogy(n[0, 1] + n[1, 1], p1)
    lr = float(-2 * (l0 - l1))
    return lr, float(chi2.sf(lr, 1))


def traffic_light_limits(T=250, p=1 - VAR_LEVEL):
    """Basel zones from the binomial distribution: green while P(X <= k) < 95%,
    yellow while < 99.99%, red after. Returns (max green, max yellow) exception counts."""
    k = np.arange(T + 1)
    cdf = binom.cdf(k, T, p)
    return int(k[cdf < 0.95].max()), int(k[cdf < 0.9999].max())


def traffic_light(hits, T=250):
    """Zone for the last T days and for the worst rolling T-day window."""
    g, y = traffic_light_limits(T)
    zone = lambda x: "green" if x <= g else "yellow" if x <= y else "red"
    c = np.convolve(np.asarray(hits, int), np.ones(T, int), "valid")
    return dict(last=int(c[-1]), last_zone=zone(c[-1]), worst=int(c.max()), worst_zone=zone(c.max()),
                worst_end=int(c.argmax() + T - 1))


def z2(x, var975, es975, alpha=1 - ES_LEVEL):
    """Acerbi-Szekely (2014) Z2 = sum(X I / (T alpha ES)) + 1, X = return. 0 if ES is right,
    negative if risk is underestimated."""
    hit = x < -var975
    return float(np.sum(x * hit / (len(x) * alpha * es975)) + 1)


def z2_test(x, f, M=2000, seed=0):
    """Z2 and its p-value, simulating Z2 under the model's own predictive distribution (H0)."""
    obs = z2(x, f["var975"], f["es975"])
    X = f["draw"](np.random.default_rng(seed), M)
    hit = X < -f["var975"]
    sims = np.sum(X * hit / (X.shape[1] * (1 - ES_LEVEL) * f["es975"]), axis=1) + 1
    return obs, float(np.mean(sims <= obs))


def backtest(r, f):
    """All tests for one model's forecasts. r: full return series."""
    x = np.asarray(r, float)[f["days"]]
    hits = x < -f["var99"]
    z, zp = z2_test(x, f)
    tl = traffic_light(hits)
    return dict(days=len(x), exceptions=int(hits.sum()), expected=len(x) * (1 - VAR_LEVEL),
                kupiec_p=kupiec(hits, 1 - VAR_LEVEL)[1], christoffersen_p=christoffersen(hits)[1],
                tl_last=f"{tl['last']} {tl['last_zone']}", tl_worst=f"{tl['worst']} {tl['worst_zone']}",
                tl_worst_end=tl["worst_end"], z2=z, z2_p=zp, hits=hits)
