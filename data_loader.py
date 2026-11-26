"""Market data access and benchmark alignment for the dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


MARKET_PRESETS: dict[str, dict[str, Any]] = {
    "Indian Market (NSE)": {
        "tickers": [
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "INFY.NS",
            "ICICIBANK.NS",
            "SBIN.NS",
            "BHARTIARTL.NS",
            "ITC.NS",
            "LT.NS",
            "BAJFINANCE.NS",
            "KOTAKBANK.NS",
            "AXISBANK.NS",
            "WIPRO.NS",
            "HCLTECH.NS",
            "SUNPHARMA.NS",
            "MARUTI.NS",
            "TITAN.NS",
            "ASIANPAINT.NS",
            "HINDUNILVR.NS",
            "ONGC.NS",
        ],
        "benchmark": "^NSEI",
        "risk_free_rate": 0.072,
        "currency": "₹",
    },
    "NASDAQ": {
        "tickers": [
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "AVGO",
            "COST",
            "NFLX",
            "AMD",
            "INTC",
            "ADBE",
            "CSCO",
            "QCOM",
            "TXN",
            "AMAT",
            "LRCX",
            "MU",
            "ISRG",
        ],
        "benchmark": "^IXIC",
        "risk_free_rate": 0.0485,
        "currency": "$",
    },
}


def _clean_history(history: pd.DataFrame) -> pd.Series:
    """Return a sorted, valid adjusted-close series from a yfinance history."""
    if history is None or history.empty:
        return pd.Series(dtype=float)
    column = "Adj Close" if "Adj Close" in history.columns else "Close"
    if column not in history.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(history[column], errors="coerce").dropna()
    series = series[series > 0]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "Adjusted Close"
    return series.astype(float)


def _latest_spot(ticker: yf.Ticker, history_prices: pd.Series) -> float:
    """Prefer the current yfinance quote, with last adjusted close as fallback."""
    try:
        fast_info = ticker.fast_info
        value = fast_info.get("last_price") if fast_info is not None else None
        spot = float(value) if value is not None else np.nan
        if np.isfinite(spot) and spot > 0:
            return spot
    except Exception:
        pass
    return float(history_prices.iloc[-1]) if not history_prices.empty else np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(
    tickers: tuple[str, ...], benchmark: str, period: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, float], dict[str, str]]:
    """Fetch adjusted closes and spot prices; cache results for one hour.

    Each symbol is fetched independently so one invalid ticker does not prevent
    the rest of the universe from loading. Historical series are aligned to the
    benchmark's available trading dates before they are returned.
    """
    requested = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    symbols = list(dict.fromkeys([*requested, benchmark.strip().upper()]))
    raw_prices: dict[str, pd.Series] = {}
    spots: dict[str, float] = {}
    errors: dict[str, str] = {}

    for symbol in symbols:
        try:
            instrument = yf.Ticker(symbol)
            history = instrument.history(period=period, auto_adjust=False, actions=False)
            prices = _clean_history(history)
            if prices.empty:
                errors[symbol] = "No historical prices returned. Check the ticker or lookback period."
                continue
            raw_prices[symbol] = prices
            spots[symbol] = _latest_spot(instrument, prices)
        except Exception as exc:  # yfinance errors vary by provider and ticker
            errors[symbol] = f"Data could not be loaded ({type(exc).__name__})."

    benchmark_symbol = benchmark.strip().upper()
    benchmark_prices = raw_prices.get(benchmark_symbol, pd.Series(dtype=float, name="Benchmark"))
    if benchmark_prices.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([])), benchmark_prices, spots, errors

    aligned: dict[str, pd.Series] = {}
    for symbol in requested:
        series = raw_prices.get(symbol)
        if series is None:
            continue
        pair = pd.concat([series.rename(symbol), benchmark_prices.rename("Benchmark")], axis=1).dropna()
        if pair.empty:
            errors[symbol] = "No overlapping trading dates with the benchmark."
            continue
        aligned[symbol] = pair[symbol]

    prices_frame = pd.DataFrame(aligned).sort_index() if aligned else pd.DataFrame(index=benchmark_prices.index)
    return prices_frame, benchmark_prices, spots, errors

