# VaR Backtest

**Which risk model would have survived 2020?** Three standard ways to compute the one-day Value at Risk and Expected Shortfall of an S&P 500 position, run day by day over 2018–2026 using only data available at the time, then judged by the same backtests regulators use.

![VaR backtest on the S&P 500](var_sp500.png)

| Model | 99% VaR exceptions (expected 20.1) | Kupiec p | Christoffersen p | Worst 250 days (Basel) | ES Z2 p | Average VaR |
|---|---|---|---|---|---|---|
| Historical simulation | 27 | 0.144 | **0.004** | **10, red** (to 2020-06) | **0.000** | 3.59% |
| EWMA normal (RiskMetrics) | **50** | **0.000** | 0.504 | **13, red** (to 2020-03) | **0.000** | 2.47% |
| Filtered HS (GARCH-t) | 22 | 0.680 | 0.239 | 7, yellow (to 2025-04) | 0.174 | 3.18% |

*p < 0.05 rejects the model. 2,013 trading days, 2018-02-08 to 2026-02-11.*

- **Historical simulation** gets the overall count right, which is why banks like it, but its exceptions arrive in clusters (Christoffersen rejects). It reacted to COVID only after the losses had happened and reached the Basel **red zone**.
- **EWMA normal** reacts fast, so its exceptions don't cluster, but normal tails are far too thin: **2.5×** the expected exceptions.
- **Filtered historical simulation** passes every test. It also holds **less** risk on average than historical simulation (3.18% vs 3.59%): it raises VaR when volatility rises and lowers it in calm markets, instead of carrying a 2-year-old crash for 2 years.

## Method

**Models** (`varbt/core.py`), each re-estimated daily on the previous 500 trading days:

| Model | Volatility | Tail shape |
|---|---|---|
| Historical simulation | none (last 500 returns as they were) | empirical |
| EWMA normal | RiskMetrics, λ = 0.94 | normal |
| Filtered HS | GARCH(1,1) with Student-t innovations, MLE refit every 21 days | empirical standardized residuals (Barone-Adesi et al.) |

**Backtests**

| Test | Question it answers |
|---|---|
| Kupiec proportion of failures | Is the number of 99% VaR exceptions right? |
| Christoffersen independence | Do exceptions cluster: is one more likely the day after another? |
| Basel traffic light | Exceptions in 250 days: green ≤ 4, yellow 5–9, red ≥ 10 (binomial 95% / 99.99%) |
| Acerbi-Szekely Z2 | Is the 97.5% Expected Shortfall (the FRTB measure) big enough? p-value by simulating under each model's own forecast distribution |

## Validation

`python tests/test_varbt.py` (or `python -m pytest`), 10 tests, about a minute. Every test checks against a known answer.

| Check | Result |
|---|---|
| GARCH variance filter vs plain recursion | equal to 1e-18 |
| GARCH-t MLE on simulated data (true α 0.10, β 0.88, ν 5) | recovers α+β within 0.015, ν within 3.5–7.5 |
| Normal 97.5% ES vs 99% VaR | 2.338σ vs 2.326σ, the Basel FRTB equivalence |
| Kupiec statistic vs binomial likelihood ratio | exact |
| Test size under a correct model | Kupiec ≈ 5.7%, Christoffersen ≈ 5%, Z2 ≈ 5.7% (target 5%) |
| Test power | clustered exceptions caught > 90%; ES too small by 30% vol caught > 85% |
| Basel zones from the binomial | green ≤ 4, red ≥ 10, matching the 1996 Basel table |
| **Known truth:** data simulated from GARCH-t, 6 seeds | FHS passes all tests on all 6; EWMA fails Kupiec on ≥ 4; HS fails Christoffersen on ≥ 3 |
| **Real data:** S&P 500 2018–2026 | results in the table above |

Two findings from building it:
- **Quantile rule matters.** NumPy's default quantile puts the 1% point of 500 scenarios about 20% too far into the body: on simulated data with a known answer, historical models broke VaR 36 times instead of ~25. The engine uses the k/(n+1) plotting position (`method="weibull"`), which fixed most of the gap.
- **Christoffersen's test is oversized in short samples:** 8% rejections instead of 5% at 1,000 days with 5% hits, converging to 5% by 5,000 days (measured). Treat borderline p-values with care.

## Run

```
pip install -r requirements.txt
python -m varbt                       # bundled S&P 500 data (FRED, 2016-2026)
python -m varbt --ticker ^GSPC        # live from Yahoo Finance, from 1990
```

Prints the table above and saves `var.png`.

## Limits

- **One asset, long only.** A portfolio needs a joint model (e.g. filtered HS on several assets with a correlation model).
- **FHS overreacts after shocks.** Its 99% VaR peaked at 27% in March 2020, when the worst daily loss was 12.8%. That conservatism is part of why it passes; a desk would also judge it on how much capital it ties up during a crisis.
- **The bundled data has no 2008.** FRED publishes 10 years of the S&P 500 index for free. `--ticker ^GSPC` adds 1990 onward, including 2008.
- **Price index, not total return.** Dividends are ignored: under 0.01% a day, small next to 1-day risk. Returns and losses are daily log returns.

Data: S&P 500 daily close from FRED via the open [datasets/s-and-p-500](https://github.com/datasets/s-and-p-500) project (ODC-PDDL); see [`tests/data`](tests/data).
