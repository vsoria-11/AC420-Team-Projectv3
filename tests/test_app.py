# [AI-GENERATED]
"""Simple tests on synthetic data (no internet needed). Run with:  pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import get_signal, run_capm, walk_forward          # noqa: E402
import sentiment                                                  # noqa: E402
import valuation                                                  # noqa: E402
from outlook import alignment_verdict, blume_beta, run_forecast, simulate_prices   # noqa: E402
from sentiment import finbert_scores, score_headlines             # noqa: E402


def make_df(beta=1.3, alpha=0.0, n=200, noise=0.01, seed=0):
    """Fake monthly returns where we KNOW the true alpha and beta."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2005-01-31", periods=n, freq="ME")
    market_excess = rng.normal(0.007, 0.04, n)
    rf = np.full(n, 0.03 / 12)
    stock_excess = alpha + beta * market_excess + rng.normal(0, noise, n)
    return pd.DataFrame({"stock": stock_excess + rf, "market": market_excess + rf, "rf": rf,
                         "stock_excess": stock_excess, "market_excess": market_excess}, index=index)


def test_capm_recovers_beta():
    capm = run_capm(make_df(beta=1.3), "Monthly")
    assert abs(capm["beta"] - 1.3) < 0.1
    assert capm["r2"] > 0.9


def test_signals():
    assert get_signal(run_capm(make_df(alpha=0.01), "Monthly"), 0.10, 0.30)[0] == "BUY"
    assert get_signal(run_capm(make_df(alpha=-0.01), "Monthly"), 0.10, 0.30)[0] == "SELL"
    assert get_signal(run_capm(make_df(alpha=0.0, seed=3), "Monthly"), 0.10, 0.30)[0] == "HOLD"


def test_low_r2_blocks_buy():
    capm = run_capm(make_df(beta=0.1, alpha=0.01, noise=0.08), "Monthly")
    assert capm["r2"] < 0.30
    assert get_signal(capm, 0.10, 0.30)[0] != "BUY"


def test_blume_beta():
    assert abs(blume_beta(2.0) - 1.67) < 1e-9
    assert abs(blume_beta(1.0) - 1.0) < 1e-9


def test_simulation_average_matches_chosen_return():
    vols = np.full(252, 0.01)
    paths = simulate_prices(100.0, 0.10, vols, n_sims=20000)
    assert paths.shape == (252, 20000)
    assert abs(paths[-1].mean() / 100 - 1.10) < 0.01


def test_forecast_runs():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 1000)))
    capm = run_capm(make_df(), "Monthly")
    result = run_forecast(prices, capm, 126)
    assert result["view"] in {"Favorable", "Mixed", "Unfavorable"}
    assert 0 <= result["prob_beat_rf"] <= 1
    assert run_forecast(prices.iloc[:100], capm, 126) is None       # too little history


def test_news_scoring_vader():
    good = [{"title": "Company beats earnings and stock surges to record high", "publisher": "A"}] * 3
    bad = [{"title": "Company faces fraud probe as shares plunge", "publisher": "B"}] * 3
    assert score_headlines(good, use_finbert=False)["flag"] == "Positive"
    assert score_headlines(bad, use_finbert=False)["flag"] == "Negative"
    assert score_headlines(good[:2], use_finbert=False)["flag"] == "Insufficient news"


class FakeFinbert:
    """Stands in for the real model: says 'positive' if the headline contains 'beats', else 'negative'."""
    def __call__(self, titles, truncation=True):
        out = []
        for title in titles:
            p = 0.9 if "beats" in title else 0.05
            n = 0.05 if "beats" in title else 0.9
            out.append([{"label": "positive", "score": p}, {"label": "negative", "score": n},
                        {"label": "neutral", "score": 1 - p - n}])
        return out


def test_finbert_scores_and_engine(monkeypatch):
    assert abs(finbert_scores(["x beats y"], FakeFinbert())[0] - 0.85) < 1e-9
    monkeypatch.setattr(sentiment, "get_finbert", lambda: FakeFinbert())
    good = [{"title": "Firm beats estimates", "publisher": "A"}] * 3
    result = score_headlines(good, use_finbert=True)
    assert result["engine"] == "FinBERT" and result["flag"] == "Positive" and result["note"] == ""


def test_falls_back_to_vader_when_finbert_unavailable(monkeypatch):
    monkeypatch.setattr(sentiment, "get_finbert", lambda: None)
    bad = [{"title": "Company faces fraud probe as shares plunge", "publisher": "B"}] * 3
    result = score_headlines(bad, use_finbert=True)
    assert result["engine"] == "VADER" and result["flag"] == "Negative" and "VADER" in result["note"]


def test_finbert_error_also_falls_back(monkeypatch):
    class Broken:
        def __call__(self, titles, truncation=True):
            raise RuntimeError("out of memory")
    monkeypatch.setattr(sentiment, "get_finbert", lambda: Broken())
    result = score_headlines([{"title": "Stock surges", "publisher": "A"}] * 3, use_finbert=True)
    assert result["engine"] == "VADER"


def test_alignment_verdict():
    kind, text = alignment_verdict("BUY", "Favorable", "Negative")
    assert kind == "error" and "Conflict" in text
    assert alignment_verdict("BUY", "Favorable", "Positive")[0] == "success"
    assert alignment_verdict("BUY", "Unfavorable", "Neutral")[0] == "warning"
    assert alignment_verdict("HOLD", "Mixed", "Negative", strong_negative=True)[0] == "warning"


def test_walk_forward():
    rows = walk_forward(make_df(n=200), window=60, horizon=6, alpha_level=0.10, r2_min=0.30)
    assert len(rows) > 10
    assert set(rows[0]) == {"date", "signal", "beta", "stock_return", "forward_alpha"}
    assert rows[0]["signal"] in {"BUY", "SELL", "HOLD"}


# ---------------------------- P/E valuation tests ----------------------------
def fake_valuation(label, pe=30.0, median=20.0):
    return {"status": "ok", "label": label, "pe": pe, "peer_median": median, "ratio": pe / median}


def test_valid_pe():
    assert valuation.valid_pe(25.5) == 25.5
    assert valuation.valid_pe(-3.0) is None
    assert valuation.valid_pe(None) is None
    assert valuation.valid_pe(float("inf")) is None


def test_classify_pe():
    assert valuation.classify_pe(1.5) == "Expensive"
    assert valuation.classify_pe(0.7) == "Cheap"
    assert valuation.classify_pe(1.0) == "In line"


def test_apply_valuation_is_downgrade_only():
    assert valuation.apply_valuation("BUY", ["r"], fake_valuation("Expensive"))[:3:2] == ("HOLD", True)
    assert valuation.apply_valuation("SELL", ["r"], fake_valuation("Cheap"))[0] == "HOLD"
    # never creates a BUY or SELL
    assert valuation.apply_valuation("HOLD", ["r"], fake_valuation("Cheap"))[0] == "HOLD"
    assert valuation.apply_valuation("HOLD", ["r"], fake_valuation("Expensive"))[0] == "HOLD"
    # unchanged cases
    assert valuation.apply_valuation("BUY", ["r"], fake_valuation("Cheap"))[0] == "BUY"
    assert valuation.apply_valuation("SELL", ["r"], fake_valuation("Expensive"))[0] == "SELL"
    unavailable = {"status": "unavailable", "label": "Not available", "message": "no P/E"}
    assert valuation.apply_valuation("BUY", [], unavailable)[0] == "BUY"


def test_get_valuation_uses_competitor_median(monkeypatch):
    pes = {"AAA": 40.0, "P1": 20.0, "P2": 22.0, "P3": 18.0, "P4": None}
    info = {"name": "AAA", "sector": "Technology", "industry": "Software", "industry_key": "software",
            "trailing_pe": 40.0, "forward_pe": 30.0}
    monkeypatch.setattr(valuation, "get_company_info", lambda ticker: info)
    monkeypatch.setattr(valuation, "find_competitors", lambda t, k, s: (["P1", "P2", "P3", "P4"], "test list"))
    monkeypatch.setattr(valuation, "get_pe", lambda ticker: pes[ticker])
    result = valuation.get_valuation("AAA")
    assert result["status"] == "ok" and result["peer_median"] == 20.0
    assert result["label"] == "Expensive"            # 40 / 20 = 2.0x
    # too few usable competitors -> check is skipped
    monkeypatch.setattr(valuation, "find_competitors", lambda t, k, s: (["P1", "P4"], "test list"))
    assert valuation.get_valuation("AAA")["status"] == "unavailable"
    # custom competitors override the automatic list
    assert valuation.get_valuation("AAA", ("P1", "P2", "P3"))["source"] == "Competitors you entered"
