# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed it,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Page 1: single-stock analyzer = CAPM signal + P/E valuation check + Forward-Looking Outlook."""

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from helpers import (DISCLAIMER, beta_label, download_prices, get_live_quote, get_signal,
                     make_returns, r2_label, run_capm, sidebar_settings)
from outlook import (alignment_verdict, fan_chart, get_analyst_view, rolling_beta_check, run_forecast)
from sentiment import get_headlines, score_headlines
from valuation import (CHEAP_RATIO, EXPENSIVE_RATIO, MAX_PEERS, apply_valuation, get_valuation)

st.set_page_config(page_title="CAPM Risk Analyzer", page_icon="📈", layout="wide")
settings = sidebar_settings()
ticker = settings["ticker"]

# Extra sidebar choices that only matter on this page
st.sidebar.divider()
st.sidebar.subheader("Valuation & news")
competitor_text = st.sidebar.text_input("Competitors for the P/E check (optional)", placeholder="e.g. MSFT, GOOGL, META",
                                        help="Leave blank to use the competitors from Yahoo Finance's industry list.")
engine_choice = st.sidebar.radio("News sentiment engine", ["FinBERT (falls back to VADER)", "VADER only"],
                                 help="FinBERT understands finance language better but is a large model. "
                                      "If it cannot load, the app switches to VADER automatically.")
custom_peers = []
for text in competitor_text.split(","):
    symbol = text.strip().upper()
    if symbol != "" and symbol != ticker and symbol not in custom_peers:
        custom_peers.append(symbol)
custom_peers = tuple(custom_peers[:MAX_PEERS])

st.title("📈 CAPM Risk Analyzer & Risk-Adjusted Return Recommendation Engine")
st.markdown("Estimate a stock's **beta** (market risk), **alpha** (return above what its risk predicts) and **R²** "
            "(how well the market explains it) to get a **CAPM signal**. A **P/E check against industry competitors** "
            "can then downgrade that signal when the price looks stretched, giving the **final recommendation**. "
            "A **Forward-Looking Outlook** adds a forecast and recent news for context.")

if ticker == "":
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

# ---- Step 1: data, regression, CAPM signal ----------------------------------------------------
try:
    prices = download_prices((ticker,), str(settings["start"]), str(settings["end"]))
    df = make_returns(prices, ticker, settings["freq"])
except ValueError as error:
    st.error(str(error))
    st.stop()

capm = run_capm(df, settings["freq"])
capm_signal, capm_reasons = get_signal(capm, settings["alpha_level"], settings["r2_min"])

# ---- Step 2: valuation check, then the FINAL recommendation ----------------------------------
with st.spinner("Checking P/E against competitors..."):
    valuation = get_valuation(ticker, custom_peers)
signal, reasons, changed = apply_valuation(capm_signal, capm_reasons, valuation)

# ---- Step 3: metric cards ---------------------------------------------------------------------
st.subheader(ticker + ": summary")
c1, c2, c3, c4, c5 = st.columns(5)

quote = get_live_quote(ticker)
if quote:
    c1.metric("Live Price", f"${quote[0]:,.2f}", f"{quote[0] / quote[1] - 1:+.2%} today",
              help="Latest price from Yahoo Finance (may be delayed).")
else:
    c1.metric("Live Price", "N/A", help="Live quote unavailable right now.")
c2.metric("Alpha (annualized)", f"{capm['alpha_annual']:+.2%}", f"p = {capm['alpha_p']:.3f}", delta_color="off",
          help="Return above (or below) what CAPM predicts for this stock's risk.")
c3.metric("Beta", f"{capm['beta']:.2f}", beta_label(capm["beta"]), delta_color="off",
          help="Sensitivity to the S&P 500. Above 1 amplifies market moves, below 1 dampens them.")
c4.metric("R²", f"{capm['r2']:.2f}", r2_label(capm["r2"]), delta_color="off",
          help="Share of the stock's movement explained by the market. Low R² means beta is unreliable.")
c5.metric("Recommendation", signal,
          f"CAPM signal: {capm_signal}" + (" (changed by P/E)" if changed else ""), delta_color="off",
          help="The CAPM signal (alpha, R², beta), adjusted by the P/E check against competitors.")

message = signal + ": " + " ".join(reasons)
{"BUY": st.success, "SELL": st.error, "HOLD": st.warning}[signal](message)

# ---- Step 4: scatter plot + CAPM return estimate ---------------------------------------------
left, right = st.columns([3, 2])
with left:
    x = df["market_excess"] * 100
    y = df["stock_excess"] * 100
    line_x = np.array([x.min(), x.max()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observations", marker=dict(opacity=0.6)))
    fig.add_trace(go.Scatter(x=line_x, y=capm["alpha"] * 100 + capm["beta"] * line_x, mode="lines",
                             name=f"OLS line: alpha={capm['alpha'] * 100:.2f}%, beta={capm['beta']:.2f}",
                             line=dict(color="red", width=3)))
    fig.update_layout(title=f"{ticker} vs S&P 500 excess returns ({settings['freq'].lower()})",
                      xaxis_title="S&P 500 excess return (%)", yaxis_title=ticker + " excess return (%)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig)
with right:
    st.markdown("#### CAPM return estimate")
    st.metric("CAPM expected return (per year)", f"{capm['expected_return']:.2%}",
              help="Risk-free rate + beta x market risk premium.")
    st.metric("Average realized return (per year)", f"{capm['realized_return']:.2%}",
              f"{capm['alpha_annual']:+.2%} vs expected", delta_color="off")
    st.write(f"Risk-free rate: {capm['rf_annual']:.2%}  |  Market risk premium: {capm['market_premium']:.2%}")
    st.write(f"Observations: {capm['n_obs']} {settings['freq'].lower()} "
             f"({df.index[0].date()} to {df.index[-1].date()})")

# ---- Step 5: valuation section -----------------------------------------------------------------
st.divider()
st.header("💲 Valuation check: P/E vs competitors")
st.caption("CAPM looks only at returns. P/E (price divided by earnings per share) shows how much investors pay for "
           "each dollar of earnings compared with the competition.")
if valuation["status"] == "ok":
    v1, v2, v3, v4 = st.columns(4)
    v1.metric(ticker + " P/E (trailing)", f"{valuation['pe']:.1f}")
    v2.metric("Competitor median P/E", f"{valuation['peer_median']:.1f}",
              help="Median of the competitors' P/E ratios (the median is not thrown off by one extreme value).")
    v3.metric("Relative P/E", f"{valuation['ratio']:.2f}x", help="Stock P/E divided by the competitor median.")
    v4.metric("Valuation", valuation["label"],
              help=f"Expensive above {EXPENSIVE_RATIO:.2f}x the median, cheap below {CHEAP_RATIO:.2f}x.")
    st.write(f"**Competitors:** {valuation['source']} ({valuation['industry'] or valuation['sector'] or 'n/a'}).")
    st.dataframe(valuation["peers"], hide_index=True, column_config={
        "P/E": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Rule: the valuation check can only downgrade. CAPM BUY + Expensive becomes HOLD, and CAPM SELL + Cheap "
               "becomes HOLD. It never creates a BUY or SELL by itself. P/E is trailing (last 12 months of earnings).")
else:
    st.info("Valuation check not used: " + valuation["message"])
    if valuation.get("peers"):
        st.dataframe(valuation["peers"], hide_index=True)

# ---- Step 6: Forward-Looking Outlook -----------------------------------------------------------
st.divider()
st.header("🔮 Forward-Looking Outlook")
st.caption("The recommendation above is built from past returns and current valuation. This panel adds a forward view "
           "and recent news. It is context only and never changes the recommendation.")

st.markdown(f"**Layer 1: Statistical signal and valuation (final recommendation): {signal}** "
            f"(CAPM signal: {capm_signal})")

# Layer 2: GARCH volatility + Monte Carlo simulation
st.markdown("**Layer 2: Quantitative forward view**")
horizon_months = st.slider("Forecast horizon (months)", 3, 24, 12)
n_days = horizon_months * 21          # about 21 trading days per month

forecast = None
try:
    recent = download_prices((ticker,), str(date.today() - timedelta(days=5 * 365)), str(date.today()))
    forecast = run_forecast(recent[ticker], capm, n_days)
except ValueError:
    pass

forecast_view = "Not available"
if forecast is None:
    st.info("Not enough daily price history (about one year needed) to build the forecast.")
else:
    forecast_view = forecast["view"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chance of beating the risk-free rate", f"{forecast['prob_beat_rf']:.0%}",
              help="Share of 5,000 simulated paths that end above what a T-bill would earn. Uses CAPM drift.")
    m2.metric("Chance of beating CAPM return if past alpha continues",
              f"{forecast['prob_beat_capm_if_alpha']:.0%}",
              help="Same simulation, but the drift also includes the stock's historical alpha. "
                   "Whether alpha really persists is exactly what the Backtest page tests.")
    m3.metric("5% Value-at-Risk", f"{forecast['var5']:+.1%}",
              help="In the worst 5% of simulated outcomes, the return over the horizon is this or lower.")
    m4.metric("Median outcome", f"{forecast['median']:+.1%}", help="The middle simulated return over the horizon.")

    f_left, f_right = st.columns([3, 2])
    with f_left:
        st.plotly_chart(fan_chart(forecast["paths"], forecast["start_price"], n_days, ticker))
    with f_right:
        st.write(f"**Forecast view: {forecast['view']}** (based on the chance of beating the risk-free rate: "
                 "60% or more = favorable, under 50% = unfavorable)")
        st.write(f"Beta: raw {capm['beta']:.2f}, Blume-adjusted {forecast['adj_beta']:.2f} "
                 "(0.67 x raw + 0.33 x 1.0, used for the forecast)")
        st.write(f"CAPM-required return with adjusted beta: {forecast['forward_return']:.2%} per year")
        st.write(f"Forecast volatility: {forecast['forecast_vol']:.1%} per year ({forecast['vol_note']}); "
                 f"historical: {capm['volatility']:.1%}")
        st.write(rolling_beta_check(df, settings["freq"])["message"])
        analyst = get_analyst_view(ticker)
        if analyst:
            st.write(f"Analyst mean target ${analyst['target']:,.2f} implies {analyst['implied_return']:+.1%} "
                     f"vs CAPM-required {forecast['forward_return']:+.1%} per year"
                     + (f" ({analyst['analysts']} analysts)." if analyst["analysts"] else "."))
        else:
            st.write("Analyst target: not available.")

# Layer 3: news sentiment
st.markdown("**Layer 3: Recent headline sentiment (context only)**")
use_finbert = engine_choice.startswith("FinBERT")
with st.spinner("Scoring headlines..."):
    news = score_headlines(get_headlines(ticker), use_finbert=use_finbert)
n1, n2, n3, n4 = st.columns(4)
n1.metric("News flag", news["flag"])
n2.metric("Average score", f"{news['average']:+.2f}" if news["count"] else "N/A",
          help="Sentiment from -1 (very negative) to +1 (very positive).")
n3.metric("Headlines used", news["count"])
n4.metric("Engine used", news["engine"])
if news["note"]:
    st.warning(news["note"])
if news["flag"] == "Insufficient news":
    st.info("Fewer than 3 recent headlines were found, so no sentiment flag is given.")
if news["rows"]:
    st.dataframe(news["rows"], hide_index=True)
st.caption("This is a snapshot of recent headlines, not a forecast. Sentiment scoring is approximate.")

# Alignment verdict (compares the FINAL recommendation with the forecast and the news)
st.markdown("**Alignment verdict (advisory)**")
kind, text = alignment_verdict(signal, forecast_view, news["flag"], news["strong_negative"])
{"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}[kind](text)

st.caption(DISCLAIMER)
