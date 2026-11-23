# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed a function,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""News sentiment: recent headlines from yfinance, scored with FinBERT (preferred) or VADER (fallback).
This is context only. Nothing here changes the Buy/Sell/Hold recommendation."""

import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FINBERT_MODEL = "ProsusAI/finbert"   # BERT model trained on financial text
MIN_HEADLINES = 3          # fewer headlines than this is not enough to say anything
POSITIVE_CUTOFF = 0.15     # average score at or above this = Positive
NEGATIVE_CUTOFF = -0.15    # at or below this = Negative
STRONG_NEGATIVE = -0.35    # at or below this = strongly negative

# VADER is built for social media, so (when it is used) we teach it a few finance words.
FINANCE_WORDS = {
    "upgrade": 2.0, "upgrades": 2.0, "downgrade": -2.0, "downgrades": -2.0,
    "beat": 1.5, "beats": 1.5, "miss": -1.5, "misses": -1.5,
    "surge": 2.0, "surges": 2.0, "soar": 2.0, "soars": 2.0, "rally": 1.5,
    "plunge": -2.5, "plunges": -2.5, "tumble": -2.0, "tumbles": -2.0, "slump": -2.0,
    "lawsuit": -1.5, "probe": -1.5, "fraud": -3.0, "bankruptcy": -3.0,
    "layoffs": -1.5, "recall": -1.5, "outperform": 2.0, "underperform": -2.0, "record": 1.0,
}


# [AI-GENERATED]
@st.cache_data(ttl=900, show_spinner=False)
def get_headlines(ticker):
    """Recent headlines from yfinance. Yahoo has changed the format over time, so handle both."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    headlines = []
    for item in items:
        content = item.get("content", item)
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider")
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else content.get("publisher", "")
        headlines.append({"title": title, "publisher": publisher})
    return headlines


# [AI-GENERATED]
@st.cache_resource(show_spinner="Loading the FinBERT model (first time only, this can take a minute)...")
def _load_finbert():
    """Load FinBERT once and keep it in memory for every user. The import is inside the function
    so the app still starts even if the transformers library is not installed."""
    from transformers import pipeline
    return pipeline("text-classification", model=FINBERT_MODEL, top_k=None)   # top_k=None = give all 3 labels


# [AI-GENERATED]
def get_finbert():
    """Return the FinBERT classifier, or None if it cannot be loaded. Never crashes the app.
    If loading fails once, we do not retry during the same session (each retry could be slow)."""
    if st.session_state.get("finbert_failed"):
        return None
    try:
        return _load_finbert()
    except Exception as error:
        st.session_state["finbert_failed"] = str(error)[:200]
        return None


# [AI-GENERATED]
def finbert_scores(titles, classifier):
    """FinBERT gives a probability for positive, negative and neutral.
    Our score = P(positive) - P(negative), which runs from -1 (very negative) to +1 (very positive)."""
    outputs = classifier(titles, truncation=True)
    if outputs and isinstance(outputs[0], dict):       # some versions return a flat list for one input
        outputs = [outputs]
    scores = []
    for labels in outputs:
        probs = {item["label"].lower(): item["score"] for item in labels}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return scores


# [AI-GENERATED]
def vader_scores(titles):
    """VADER fallback: a dictionary-based scorer (-1 to +1) with a few finance words added."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCE_WORDS)
    return [analyzer.polarity_scores(title)["compound"] for title in titles]


# [AI-GENERATED]
def score_headlines(headlines, use_finbert=True):
    """Score every headline, average the scores, and turn the average into a flag.
    Tries FinBERT first; if it is unavailable or fails, falls back to VADER and says so."""
    titles = [headline["title"] for headline in headlines]
    scores = None
    engine = "VADER"
    note = ""

    if use_finbert and titles:
        classifier = get_finbert()
        if classifier is not None:
            try:
                scores = finbert_scores(titles, classifier)
                engine = "FinBERT"
            except Exception:
                scores = None
        if scores is None:
            note = "FinBERT could not be loaded or run here, so VADER was used instead."
    if scores is None:
        scores = vader_scores(titles)

    rows = []
    for headline, score in zip(headlines, scores):
        rows.append({"Headline": headline["title"], "Publisher": headline["publisher"], "Score": round(score, 2)})

    count = len(rows)
    average = sum(row["Score"] for row in rows) / count if count else 0.0
    if count < MIN_HEADLINES:
        flag = "Insufficient news"
    elif average >= POSITIVE_CUTOFF:
        flag = "Positive"
    elif average <= NEGATIVE_CUTOFF:
        flag = "Negative"
    else:
        flag = "Neutral"
    return {"flag": flag, "average": average, "count": count, "rows": rows, "engine": engine, "note": note,
            "strong_negative": count >= MIN_HEADLINES and average <= STRONG_NEGATIVE}


# ---------------------------------------------------------------------------
# [TEAM-WRITTEN] space: put code your team writes yourselves below this line.
# ---------------------------------------------------------------------------
