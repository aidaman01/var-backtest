"""Every test checks against a known answer. Run: python -m pytest  (or python tests/test_varbt.py)"""
import os
import sys

import numpy as np
from scipy.stats import binom, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from varbt.core import (_tail, backtest, christoffersen, forecast, garch_fit, garch_variance, kupiec,
                        traffic_light_limits, z2_test)
from varbt.data import load_csv, simulate_garch_t

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sp500_fred.csv")


def test_garch_filter_matches_recursion():
    e = np.random.default_rng(1).standard_normal(200) * 0.01
    s = [2e-4]
    for x in e:
        s.append(1e-6 + 0.1 * x * x + 0.85 * s[-1])
    assert np.abs(garch_variance(e, 1e-6, 0.1, 0.85, 2e-4) - s).max() < 1e-18


def test_garch_mle_recovers_parameters():
    for seed in (3, 4):
        r, _ = simulate_garch_t(5000, a=0.10, b=0.88, nu=5.0, seed=seed)
        mu, omega, a, b, nu, g = garch_fit(r)
        assert g == 0.0
        assert abs(a - 0.10) < 0.03 and abs(b - 0.88) < 0.03 and abs(a + b - 0.98) < 0.015, (a, b)
        assert 3.5 < nu < 7.5, nu


def test_gjr_mle_recovers_parameters():
    # Known truth a 0.03, g 0.12, b 0.88 (persistence a + g/2 + b = 0.97), nu 5. Over 12 seeds the
    # estimates had sd ~0.01 (a, g, b) and ~0.25 (nu): tolerances are ~3 sd.
    for seed in (3, 4):
        r, _ = simulate_garch_t(5000, a=0.03, g=0.12, b=0.88, nu=5.0, seed=seed)
        mu, omega, a, b, nu, g = garch_fit(r, gjr=True)
        assert abs(g - 0.12) < 0.04 and abs(a - 0.03) < 0.03 and abs(a + g / 2 + b - 0.97) < 0.03, (a, g, b)
        assert 3.5 < nu < 7.5, nu
    r, _ = simulate_garch_t(5000, a=0.10, b=0.88, nu=5.0, seed=3)  # symmetric truth: no leverage found
    assert abs(garch_fit(r, gjr=True)[5]) < 0.04


def test_garch_mle_reaches_optimum_on_real_windows():
    # Windows where a+b sits near 0.999: the old penalty-wall fit stalled 3-6 log-lik units short.
    # Known answer: a long Nelder-Mead search on the textbook likelihood can't beat garch_fit.
    from scipy.optimize import minimize
    from scipy.stats import t as student_t
    r = load_csv(DATA).values

    def nll(p, x):
        mu, lw, a, b, nu = p
        if a < 0 or b < 0 or a + b >= 0.999 or not 2.1 <= nu <= 50:
            return 1e10
        e = x - mu
        s2 = garch_variance(e, np.exp(lw), a, b, x.var())[:-1]
        k = np.sqrt(nu / (nu - 2))
        return -np.sum(student_t.logpdf(e / np.sqrt(s2) * k, nu) + np.log(k) - 0.5 * np.log(s2))

    for t0 in (1949, 1991, 2033):
        x = r[t0 - 500:t0]
        mu, omega, a, b, nu, _ = garch_fit(x)
        p = [mu, np.log(omega), a, b, nu]
        best = min(minimize(nll, x0, args=(x,), method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-11}).fun
                   for x0 in (p, [x.mean(), np.log(x.var() * 0.02), 0.15, 0.83, 5.0]))
        assert nll(p, x) - best < 0.05, (t0, nll(p, x), best)


def test_normal_var_es_match_basel_constants():
    # Basel's FRTB swap of 99% VaR for 97.5% ES: under normality 2.326 sigma vs 2.338 sigma.
    f = forecast(np.random.default_rng(0).standard_normal(600) * 0.01, model="ewma")
    sig = f["var99"] / norm.ppf(0.99)
    assert np.allclose(f["es975"] / sig, 2.3378, atol=1e-4)


def test_historical_tail_on_known_sample():
    S = -np.arange(1.0, 501.0)[None, :]          # losses 1..500
    var, es = _tail(S, 0.99)
    assert np.isclose(var[0], 495.99) and np.isclose(es[0], 498.0)  # k/(n+1) position 5.01; mean of 496..500


def test_kupiec_matches_binomial_likelihood_ratio():
    hits = np.zeros(250, bool)
    hits[:10] = True
    lr, _ = kupiec(hits, 0.01)
    assert np.isclose(lr, 2 * (binom.logpmf(10, 250, 10 / 250) - binom.logpmf(10, 250, 0.01)))


def test_kupiec_and_christoffersen_size_and_power():
    rng = np.random.default_rng(0)
    size_k = np.mean([kupiec(rng.random(1000) < 0.01, 0.01)[1] < 0.05 for _ in range(2000)])
    size_c = np.mean([christoffersen(rng.random(5000) < 0.05)[1] < 0.05 for _ in range(1000)])
    assert 0.035 < size_k < 0.075 and 0.035 < size_c < 0.07, (size_k, size_c)

    def clustered(T=1000, p01=0.04, p11=0.3):
        h = np.zeros(T, bool)
        for t in range(1, T):
            h[t] = rng.random() < (p11 if h[t - 1] else p01)
        return h

    assert np.mean([christoffersen(clustered())[1] < 0.05 for _ in range(200)]) > 0.9


def test_basel_traffic_light_limits():
    assert traffic_light_limits() == (4, 9)  # green 0-4, yellow 5-9, red 10+ (Basel 1996 table)


def test_es_z2_size_and_power():
    rng, T = np.random.default_rng(0), 250
    sig = np.full(T, 0.01)
    f = dict(var975=-sig * norm.ppf(0.025), es975=sig * norm.pdf(norm.ppf(0.025)) / 0.025,
             draw=lambda g, M: sig * g.standard_normal((M, T)))
    size = np.mean([z2_test(sig * rng.standard_normal(T), f, M=1000, seed=i)[1] < 0.05 for i in range(300)])
    power = np.mean([z2_test(1.3 * sig * rng.standard_normal(T), f, M=1000, seed=i)[1] < 0.05 for i in range(200)])
    assert 0.025 < size < 0.085 and power > 0.85, (size, power)


def test_models_on_simulated_garch_t():
    """Data from GARCH(1,1)-t, so FHS is the right model. It should pass; the others should not."""
    res = {m: [] for m in ("hs", "ewma", "fhs")}
    for seed in range(6):
        r, _ = simulate_garch_t(3000, seed=seed)
        for m in res:
            res[m].append(backtest(r, forecast(r, model=m)))
    assert all(b["kupiec_p"] > 0.05 and b["christoffersen_p"] > 0.05 and b["z2_p"] > 0.05 for b in res["fhs"])
    assert sum(b["kupiec_p"] < 0.05 for b in res["ewma"]) >= 4            # normal tails too thin
    assert sum(b["christoffersen_p"] < 0.05 for b in res["hs"]) >= 3       # exceptions cluster


def test_real_sp500():
    """FRED S&P 500, 2016-02 to 2026-02 (tests/data). Backtest 2018-02 to 2026-02, 2013 days."""
    r = load_csv(DATA)
    out = {m: backtest(r.values, forecast(r.values, model=m)) for m in ("hs", "ewma", "fhs")}
    hs, ewma, fhs = out["hs"], out["ewma"], out["fhs"]
    assert fhs["kupiec_p"] > 0.05 and fhs["christoffersen_p"] > 0.05 and fhs["z2_p"] > 0.05
    assert not fhs["tl_worst"].endswith("red")
    assert hs["christoffersen_p"] < 0.05 and hs["tl_worst"].endswith("red")   # clustered, COVID
    assert r.index[500 + hs["tl_worst_end"]].year == 2020
    assert ewma["kupiec_p"] < 0.01 and ewma["exceptions"] > 2 * ewma["expected"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
