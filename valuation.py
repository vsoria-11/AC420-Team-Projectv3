# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed a function,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Valuation check: compare a stock's P/E ratio with the P/E of its industry competitors.
CAPM only looks at returns. P/E adds a look at the PRICE: are we paying more or less per dollar of
earnings than for the competition? The result can only DOWNGRADE a CAPM signal (see apply_valuation)."""

import numpy as np
import streamlit as st
import yfinance as yf

EXPENSIVE_RATIO = 1.30   # stock P/E more than 30% above the competitor median = Expensive
CHEAP_RATIO = 0.80       # more than 20% below the competitor median = Cheap
MIN_PEERS = 3            # need at least this many competitors with a usable P/E
MAX_PEERS = 8

# Backup competitor lists (large companies by Yahoo Finance sector), used only if Yahoo's
# industry list cannot be fetched.
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "INTC"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MDLZ", "CL"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA"],
    "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OXY"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "GE", "RTX", "LMT", "DE"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "O", "WELL", "SPG", "CCI"],
    "Basic Materials": ["LIN", "SHW", "APD", "FCX", "NEM", "ECL", "DD", "NUE"],
}


# [AI-GENERATED]
def valid_pe(value):
    """A P/E is only meaningful if it is a positive, finite number (loss-making companies have none)."""
    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
        return float(value)
    return None


# [AI-GENERATED]
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_company_info(ticker):
    """Name, sector, industry and P/E from Yahoo Finance (cached for 6 hours). None if unavailable."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    if not info:
        return None
    return {"name": info.get("shortName", ticker), "sector": info.get("sector"),
            "industry": info.get("industry"), "industry_key": info.get("industryKey"),
            "trailing_pe": info.get("trailingPE"), "forward_pe": info.get("forwardPE")}


# [AI-GENERATED]
def get_pe(ticker):
    """Trailing P/E (price / last 12 months of earnings per share), or None if not meaningful."""
    info = get_company_info(ticker)
    if info is None:
        return None
    return valid_pe(info["trailing_pe"])


# [AI-GENERATED]
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def find_competitors(ticker, industry_key, sector):
    """Find competitors: first Yahoo Finance's list of top companies in the same INDUSTRY,
    then (if that fails) a backup list for the same SECTOR."""
    peers = []
    source = ""
    if industry_key:
        try:
            top = yf.Industry(industry_key).top_companies      # table indexed by ticker symbol
            peers = [symbol for symbol in top.index if symbol != ticker]
            source = "Top companies in the same Yahoo Finance industry"
        except Exception:
            peers = []
    if len(peers) < MIN_PEERS:
        peers = [symbol for symbol in SECTOR_PEERS.get(sector, []) if symbol != ticker]
        source = "Backup list of large companies in the same sector"
    return peers[:MAX_PEERS], source


# [AI-GENERATED]
def classify_pe(ratio):
    """ratio = stock P/E divided by the competitors' median P/E."""
    if ratio > EXPENSIVE_RATIO:
        return "Expensive"
    if ratio < CHEAP_RATIO:
        return "Cheap"
    return "In line"


# [AI-GENERATED]
def get_valuation(ticker, custom_peers=()):
    """Compare the stock's P/E with its competitors' median P/E. Returns a dictionary.
    status == "ok" when the check could be done, otherwise "unavailable" with a message."""
    unavailable = {"status": "unavailable", "label": "Not available"}

    info = get_company_info(ticker)
    if info is None:
        return {**unavailable, "message": "Company data was not available from Yahoo Finance."}
    stock_pe = valid_pe(info["trailing_pe"])
    if stock_pe is None:
        return {**unavailable, "message": "This stock has no meaningful P/E (for example, it has negative earnings)."}

    if len(custom_peers) > 0:
        peers, source = list(custom_peers), "Competitors you entered"
    else:
        peers, source = find_competitors(ticker, info["industry_key"], info["sector"])

    rows = []
    for peer in peers:
        rows.append({"Ticker": peer, "P/E": get_pe(peer)})       # None when the peer has no usable P/E
    usable = [row["P/E"] for row in rows if row["P/E"] is not None]
    if len(usable) < MIN_PEERS:
        return {**unavailable, "message": f"Only {len(usable)} competitors had a usable P/E (need {MIN_PEERS}).",
                "peers": rows, "source": source}

    median = float(np.median(usable))          # the median is not thrown off by one extreme P/E
    ratio = stock_pe / median
    return {"status": "ok", "label": classify_pe(ratio), "pe": stock_pe, "forward_pe": valid_pe(info["forward_pe"]),
            "peer_median": median, "ratio": ratio, "peers": rows, "source": source,
            "sector": info["sector"], "industry": info["industry"]}


# [AI-GENERATED]
def apply_valuation(signal, reasons, valuation):
    """Turn the CAPM signal into the FINAL recommendation. Downgrade-only:
       CAPM BUY  + Expensive -> HOLD   (alpha looks good, but the price already looks stretched)
       CAPM SELL + Cheap     -> HOLD   (the weak results may already be in a low price)
    P/E can never create a BUY or SELL by itself. Returns (final_signal, reasons, changed)."""
    reasons = list(reasons)
    if valuation["status"] != "ok":
        reasons.append("Valuation check not used: " + valuation["message"])
        return signal, reasons, False

    text = (f"P/E {valuation['pe']:.1f} vs competitor median {valuation['peer_median']:.1f} "
            f"({valuation['ratio']:.2f}x)")
    label = valuation["label"]

    if signal == "BUY" and label == "Expensive":
        reasons.append("Downgraded from BUY to HOLD by the valuation check: " + text
                       + ", so the price already looks stretched compared with competitors.")
        return "HOLD", reasons, True
    if signal == "SELL" and label == "Cheap":
        reasons.append("Softened from SELL to HOLD by the valuation check: " + text
                       + ", so the weak results may already be reflected in a low price.")
        return "HOLD", reasons, True

    reasons.append("Valuation check (" + label.lower() + "): " + text + ". The CAPM signal is unchanged.")
    return signal, reasons, False


# ---------------------------------------------------------------------------
# [TEAM-WRITTEN] space: put code your team writes yourselves below this line.
# ---------------------------------------------------------------------------
