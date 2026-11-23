# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed it,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Page 5: how the recommendation is decided + AI-use statement (requested by the professor)."""

import streamlit as st

from helpers import DISCLAIMER
from valuation import CHEAP_RATIO, EXPENSIVE_RATIO, MIN_PEERS

st.set_page_config(page_title="How It Works & AI Use", page_icon="📘", layout="wide")

# ==========================================================================================
# EDIT THIS BLOCK so the AI-use statement matches what your team really did.
# Anything still in [BRACKETS] shows on the page as a reminder.
# ==========================================================================================
AI_TOOL = "Claude (Anthropic)"
AI_GENERATED = [
    "The Python and Streamlit code for the app: data download, CAPM regression, decision logic, charts, "
    "forecast, backtest, and pages",
    "The P/E valuation module (finding competitors, comparing P/E with the competitor median, and the downgrade rule)",
    "The news-sentiment module, including the FinBERT integration and the VADER fallback",
    "The unit tests, the deployment test file and the README",
    "A first draft of the text on this page",
    "[EDIT: add or remove items so this list is accurate]",
]
TEAM_CONTRIBUTIONS = [
    "The project proposal and specifications",
    "Choosing the model (single-factor CAPM regression) and the data sources",
    "Defining the CAPM Buy / Sell / Hold decision rules and thresholds",
    "Directing the work, testing the app and checking the results",
    "The decision to add the news-sentiment layer as context only",
    "Acting on our professor's feedback: choosing FinBERT instead of VADER, and choosing the P/E ratio, compared "
    "with industry competitors, as the way to bring the stock's price into the decision",
    "[EDIT: confirm the team reviewed and approved the P/E thresholds and the downgrade-only rule]",
    "[EDIT: add anything else your team did, e.g. code you wrote or rewrote yourselves]",
]
CLASS_NOTE = ("[CONFIRM BEFORE SUBMITTING: Our team covered the basic topics (regression, returns, "
              "Streamlit) in class, and we reviewed the code and can explain it.]")
# ==========================================================================================

st.title("📘 How It Works & AI Use")

tab1, tab2, tab3 = st.tabs(["How the recommendation is decided", "Limitations", "AI-use statement"])

with tab1:
    st.header("From data to a recommendation")
    st.markdown("""
**1. Data.** Daily prices for the stock, the S&P 500 (^GSPC) and the 13-week T-bill yield (^IRX) come from Yahoo
Finance (yfinance). Prices become returns, and subtracting the risk-free rate gives **excess returns**.

**2. CAPM regression.** We fit a straight line through the stock's excess returns versus the market's excess returns
(ordinary least squares, OLS):
""")
    st.latex(r"R_i - R_f = \alpha_i + \beta_i (R_M - R_f) + \varepsilon_i")
    st.markdown("""
- **Beta** is the slope: how strongly the stock moves with the market (market risk).
- **Alpha** is the intercept: return above or below what the stock's risk predicts.
- **R²** is how much of the stock's movement the market explains (how far to trust beta).
- **p-values** tell us whether alpha is statistically different from zero or just noise.

**3. CAPM signal (numbers only).**

| CAPM signal | Rule |
|---|---|
| **BUY** | Alpha above zero, statistically significant, and R² at or above the threshold |
| **SELL** | Alpha below zero and significant, **or** beta above 1.5 with zero or negative average excess return |
| **HOLD** | Alpha within its standard error of zero, or R² below the threshold |

The significance level and R² threshold are set in the sidebar.

**4. Valuation check: P/E compared with industry competitors.** CAPM looks only at returns and ignores the *price*
we pay. The P/E ratio (share price divided by earnings per share) shows how much investors pay for each dollar of
earnings. A P/E only means something relative to similar companies, so we compare it with competitors:
""")
    st.markdown(f"""
- **Competitors:** the top companies in the same Yahoo Finance industry (up to 8). If that list cannot be fetched,
  we use a backup list of large companies in the same sector. You can also type your own competitors in the sidebar.
- **Comparison:** the stock's trailing P/E divided by the **median** P/E of its competitors (the median is not thrown
  off by one extreme value). At least {MIN_PEERS} competitors with a usable P/E are needed.
- **Label:** *Expensive* if the ratio is above {EXPENSIVE_RATIO:.2f}x, *Cheap* if below {CHEAP_RATIO:.2f}x, otherwise *In line*.
- **If there is no usable P/E** (for example, a company with negative earnings), the check is skipped and says so.

**5. Final recommendation.** The valuation check can only **downgrade** the CAPM signal. It can never create a BUY or
SELL by itself.

| CAPM signal | Valuation | Final recommendation |
|---|---|---|
| BUY | Expensive | **HOLD** (good alpha, but the price already looks stretched) |
| SELL | Cheap | **HOLD** (weak results may already be in a low price) |
| Any other combination | Any | Same as the CAPM signal |

**6. Forward-looking outlook (context, never changes the recommendation).**
- *Forecast volatility* with a GARCH(1,1) model, which lets volatility rise and fall over time.
- *Monte Carlo simulation* of 5,000 possible price paths using the CAPM-implied return (with a Blume-adjusted beta,
  0.67 x raw + 0.33 x 1). From it we read the chance of beating the risk-free rate, the 5% Value-at-Risk and a fan chart.
- *Second scenario:* the same simulation with the historical alpha added, to ask "what if past alpha continues?"
- *Analyst price target* from Yahoo Finance compared with the CAPM-required return, when available.
- *Forecast view:* Favorable if the chance of beating the risk-free rate is 60% or more, Unfavorable if under 50%,
  otherwise Mixed.

**7. News layer (context only).** Recent headlines from yfinance are scored with **FinBERT**, a language model trained
on financial text, so it reads context better than a word-list tool. Each headline gets a score from -1 (negative) to
+1 (positive): the model's probability of "positive" minus its probability of "negative". FinBERT is a large model, so
if it cannot be loaded in the hosting environment, the app **automatically falls back to VADER** (a simpler
dictionary-based scorer with some finance words added) and tells you which engine was used. The average score becomes
Positive, Neutral or Negative. With fewer than 3 headlines we show "Insufficient news". It is a snapshot of recent
headlines, not a forecast.

**8. Alignment verdict (advisory).** A short sentence compares the final recommendation, the forecast and the news.
For example, BUY with negative news is shown as a **conflict** to investigate before buying; BUY with a favorable
forecast and positive news is **strong alignment**. The verdict never changes the recommendation.

**9. Backtest.** The Backtest page tests the **CAPM signal** on data the model has not seen, by fitting on a trailing
window and checking the next 3, 6 or 12 months. The P/E adjustment, forecast and news are not backtested (see
Limitations).
""")

with tab2:
    st.header("Limitations")
    st.markdown("""
- **Single factor.** CAPM ignores other known drivers of returns such as size, value and momentum.
- **Proxies.** The S&P 500 stands in for the market portfolio, and the 13-week T-bill yield stands in for the risk-free rate.
- **Window dependence.** Alpha and beta change with the date range and the return frequency.
- **Noisy alpha.** Over short windows most alphas are not statistically significant, so many stocks get HOLD.
- **P/E limits.** Trailing P/E uses the last 12 months of earnings, which can be distorted by one-off items.
  Companies with negative earnings have no P/E. Even within an industry, competitors can differ in size, growth and
  business mix, and a high P/E sometimes reflects high expected growth rather than overpricing. The competitor list
  comes from Yahoo Finance and can be incomplete. The thresholds (1.30x and 0.80x) are judgment calls.
- **P/E is not backtested.** yfinance does not provide the P/E a stock or its competitors had on past dates, so the
  Backtest page tests the CAPM signal only. The final recommendation, which includes P/E, has not been tested
  out-of-sample.
- **Forecast assumptions.** The simulation assumes normally distributed shocks and a constant expected return, which is a
  simplification. The "past alpha continues" scenario assumes something the backtest may not support.
- **News is a snapshot.** FinBERT reads financial text better than a word list, but headline sentiment is still
  approximate, covers only recent headlines and cannot be backtested. If FinBERT cannot run in the hosting
  environment, the simpler VADER fallback is used, which is less accurate on context and sarcasm.
- **Backtest caveats.** Results may be mixed, samples are small, stocks are chosen from today's survivors, and trading
  costs are ignored.
- **Data source.** The original proposal mentioned WRDS/CRSP data (1990-2025). That data is licensed and cannot be
  redistributed on a public repository, so the app uses yfinance.
- **Not investment advice.** This is an educational project.
""")

with tab3:
    st.header("AI-use statement")
    st.markdown("**AI tool used:** " + AI_TOOL)
    st.markdown("**What the AI generated:**")
    for item in AI_GENERATED:
        st.markdown("- " + item)
    st.markdown("**What our team contributed:**")
    for item in TEAM_CONTRIBUTIONS:
        st.markdown("- " + item)
    st.markdown("**Understanding of the code:** " + CLASS_NOTE)
    st.markdown("**Code tags in the GitHub repository:** every file and function starts with a comment tag. "
                "`# [AI-GENERATED]` = written by AI, not yet reviewed by the team. "
                "`# [AI-GENERATED, TEAM-REVIEWED]` = written by AI and reviewed by the team. "
                "`# [TEAM-WRITTEN]` = written by the team.")

st.caption(DISCLAIMER)
