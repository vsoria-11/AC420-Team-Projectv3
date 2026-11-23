# CAPM Risk Analyzer & Risk-Adjusted Return Recommendation Engine

AC 420 Intro to Financial Analytics. A multi-page **Streamlit** app that measures a stock's market risk with a
single-factor **CAPM regression**, issues a **CAPM Buy / Sell / Hold signal**, adjusts it with a **P/E check against
industry competitors**, and adds a **forward-looking outlook** (forecast + news context) and an **out-of-sample
backtest** to back the signal up.

```
(R_i - R_f) = alpha_i + beta_i (R_M - R_f) + error_i
```

## AI-tag legend (read this first)

Every file and function starts with a comment tag that says who wrote it:

| Tag | Meaning |
|---|---|
| `# [AI-GENERATED]` | Written with AI (Claude); not yet reviewed by the team |
| `# [AI-GENERATED, TEAM-REVIEWED]` | Written with AI; the team has read it and can explain it |
| `# [TEAM-WRITTEN]` | Written by the team |

Each helper file ends with a `[TEAM-WRITTEN]` space for code the team adds. Update tags only to match what really
happened. The "How It Works & AI Use" page contains the full AI-use statement.

## How the recommendation is built

1. **CAPM signal** (`helpers.py`): BUY / SELL / HOLD from alpha significance, R² and beta.
2. **Valuation check** (`valuation.py`): the stock's trailing P/E is compared with the **median P/E of its industry
   competitors** (top companies in the same Yahoo Finance industry, a backup sector list, or competitors you type in).
   *Expensive* = more than 1.30x the median, *Cheap* = less than 0.80x.
3. **Final recommendation:** the valuation check can only **downgrade**: CAPM BUY + Expensive becomes HOLD, and
   CAPM SELL + Cheap becomes HOLD. It never creates a BUY or SELL by itself.
4. **Forward-Looking Outlook** (`outlook.py`, `sentiment.py`), **context only, never changes the recommendation:**
   GARCH(1,1) volatility + Monte Carlo simulation (fan chart, chance of beating the risk-free rate, 5% Value-at-Risk),
   analyst target vs CAPM return, rolling-beta note, and news sentiment scored with **FinBERT** (automatic fallback
   to **VADER** if FinBERT cannot run). An alignment verdict compares the final recommendation with the forecast and news.

### CAPM decision matrix

| CAPM signal | Rule |
|---|---|
| **BUY** | Alpha > 0, statistically significant, and R² >= threshold |
| **SELL** | Alpha < 0 and significant, **or** beta > 1.5 with average excess return <= 0 |
| **HOLD** | Alpha within its standard error of zero, or R² below threshold |

## Pages

| Page | What it does |
|---|---|
| **Home** | Ticker + date-range slider; cards for live price, alpha, beta, R² and the final recommendation; scatter plot; **P/E vs competitors**; **Forward-Looking Outlook** |
| **Peer Risk Comparison** | Multi-select stocks; beta and R² bars vs the S&P 500; Security Market Line; Capital Market Line; P/E column; CSV download |
| **Model Diagnostics** | Full OLS table (coefficients, standard errors, t-stats, p-values); residual plots; plain-English explanations |
| **Backtest** | Walk-forward test of the **CAPM signal**: fit on a trailing window, then measure the next 3 / 6 / 12 months |
| **How It Works & AI Use** | How the recommendation is decided, limitations, AI-use statement |

## Project structure

```
Home.py                            # Page 1 (entry point): signal, P/E check, outlook
pages/
  2_Peer_Risk_Comparison.py
  3_Model_Diagnostics.py
  4_Backtest.py
  5_How_It_Works_and_AI_Use.py
helpers.py                         # sidebar, data download, returns, CAPM, CAPM signal, walk-forward backtest
valuation.py                       # P/E vs competitors, downgrade-only rule -> final recommendation
outlook.py                         # GARCH, Monte Carlo, analyst view, alignment verdict
sentiment.py                       # headlines, FinBERT scoring, VADER fallback
finbert_test.py                    # tiny app to test FinBERT on Streamlit Cloud (deploy separately)
tests/test_app.py                  # unit tests on synthetic data (no internet needed)
requirements.txt                   # includes FinBERT (torch CPU + transformers)
requirements-no-finbert.txt        # backup without FinBERT
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py
```

Tests: `pip install -r requirements-dev.txt && pytest` (does not download FinBERT).

## Deploy to Streamlit Community Cloud, with FinBERT

FinBERT (~440 MB model plus PyTorch) is large, and Community Cloud has limited memory (roughly 2.7 GB at the time of
writing; limits can change). **Test it before relying on it:**

1. Push the repo to GitHub.
2. On <https://share.streamlit.io>, first deploy **`finbert_test.py`** as its own app. Click the button and check the
   load time and the peak memory it reports.
3. If it works, deploy the main app with main file **`Home.py`**.
4. If the deployment fails or the app runs out of memory, replace the contents of `requirements.txt` with
   `requirements-no-finbert.txt` and redeploy. The app detects that FinBERT is missing and uses VADER automatically.
   You can also choose "VADER only" in the Home sidebar at any time.
5. Expect the first FinBERT use after the app wakes up (apps sleep after 12 hours without visitors) to be slow, because
   the model has to load again. Open the app a few minutes before presenting.

## Data notes

- **Source:** yfinance (adjusted closes, P/E, news, analyst targets). Market = S&P 500 (`^GSPC`); risk-free proxy =
  13-week T-bill yield (`^IRX`).
- **WRDS:** the original proposal mentioned WRDS/CRSP/Compustat data (1990-2025). That data is licensed and cannot
  be redistributed in a public repo, so the app uses yfinance. The date slider reaches back to 1990.
- **Competitors for P/E:** Yahoo Finance's industry list (up to 8 companies), a backup list by sector, or tickers you
  type in the Home sidebar. At least 3 competitors with a usable P/E are needed, otherwise the check is skipped.
- **Settings across pages:** the sidebar values are kept when you switch pages by re-saving each key in
  `st.session_state` at the top of every page (see `sidebar_settings` in `helpers.py`).

## Limitations

Single factor; index and T-bill proxies; estimates depend on window and frequency; P/E uses trailing earnings and
imperfect competitor lists; P/E and news cannot be backtested (no historical data in yfinance), so the backtest covers
the CAPM signal only; simulation assumes normal shocks; headline sentiment is approximate; backtest has small samples,
survivorship bias and no trading costs. Educational project, not investment advice.
