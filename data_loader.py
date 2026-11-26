"""Market data access and benchmark alignment for the dashboard."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from io import BytesIO
from threading import RLock
from time import monotonic
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


def _ttl_cache(seconds: int, maxsize: int = 64):
    """Small thread-safe TTL cache shared by the Streamlit app and API process."""
    def decorate(function):
        cache: OrderedDict[tuple[Any, ...], tuple[float, Any]] = OrderedDict()
        lock = RLock()

        @wraps(function)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = monotonic()
            with lock:
                entry = cache.get(key)
                if entry is not None and now - entry[0] < seconds:
                    cache.move_to_end(key)
                    return deepcopy(entry[1])
            value = function(*args, **kwargs)
            with lock:
                cache[key] = (monotonic(), value)
                cache.move_to_end(key)
                while len(cache) > maxsize:
                    cache.popitem(last=False)
            return deepcopy(value)

        return wrapped

    return decorate


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
            "ABNB",
            "ADBE",
            "ADI",
            "ADP",
            "ADSK",
            "AEP",
            "ALAB",
            "ALNY",
            "AMAT",
            "AMD",
            "AMGN",
            "AMZN",
            "APP",
            "ARM",
            "ASML",
            "AVGO",
            "AXON",
            "BKNG",
            "BKR",
            "CCEP",
            "CDNS",
            "CEG",
            "CMCSA",
            "COST",
            "CPRT",
            "CRWD",
            "CRWV",
            "CSCO",
            "CSX",
            "CTAS",
            "DASH",
            "DDOG",
            "DXCM",
            "EXC",
            "FANG",
            "FAST",
            "FER",
            "FTNT",
            "GEHC",
            "GILD",
            "GOOG",
            "GOOGL",
            "HON",
            "HONA",
            "IDXX",
            "INTC",
            "INTU",
            "ISRG",
            "KDP",
            "KHC",
            "KLAC",
            "LIN",
            "LITE",
            "LRCX",
            "MAR",
            "MCHP",
            "MDLZ",
            "MELI",
            "META",
            "MNST",
            "MPWR",
            "MRVL",
            "MSFT",
            "MSTR",
            "MU",
            "NBIS",
            "NFLX",
            "NVDA",
            "NXPI",
            "ODFL",
            "ORLY",
            "PANW",
            "PAYX",
            "PCAR",
            "PDD",
            "PEP",
            "PLTR",
            "PYPL",
            "QCOM",
            "REGN",
            "RKLB",
            "ROP",
            "ROST",
            "SBUX",
            "SHOP",
            "SNDK",
            "SNPS",
            "SPCX",
            "STX",
            "TER",
            "TMUS",
            "TRI",
            "TSLA",
            "TTWO",
            "TXN",
            "VRTX",
            "WBD",
            "WDAY",
            "WDC",
            "WMT",
            "XEL",
        ],
        "benchmark": "^IXIC",
        "risk_free_rate": 0.0485,
        "currency": "$",
    },
}

NIFTY500_CONSTITUENTS_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
NIFTY500_FALLBACK = tuple(MARKET_PRESETS["Indian Market (NSE)"]["tickers"])
# Yahoo does not consistently publish new Indian listings under their NSE
# suffix. BAGMANE is available under its BSE listing alias instead.
YFINANCE_TICKER_FALLBACKS = {"BAGMANE.NS": ("BAGMANE.BO", "544758.BO")}


@_ttl_cache(seconds=24 * 60 * 60)
def load_nifty500_tickers() -> tuple[tuple[str, ...], str]:
    """Load the current NIFTY 500 constituents, falling back to the core NSE list."""
    try:
        request = Request(
            NIFTY500_CONSTITUENTS_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; StockRiskReturnDashboard/1.0)",
                "Referer": "https://www.niftyindices.com/",
            },
        )
        with urlopen(request, timeout=15) as response:
            constituent_file = response.read()
        constituents = pd.read_csv(BytesIO(constituent_file))
        symbol_column = next((column for column in constituents.columns if str(column).strip().lower() == "symbol"), None)
        if symbol_column is None:
            raise ValueError("The constituent file has no Symbol column")
        symbols = constituents[symbol_column].dropna().astype(str).str.strip().str.upper()
        tickers = tuple(
            dict.fromkeys(
                f"{symbol}.NS"
                for symbol in symbols
                if symbol and symbol != "NAN" and not symbol.startswith("DUMMY")
            )
        )
        if len(tickers) < 400:
            raise ValueError(f"Only {len(tickers)} constituents were returned")
        return tickers, ""
    except Exception as exc:
        return NIFTY500_FALLBACK, (
            "The current NIFTY 500 list was unavailable; using the built-in NSE fallback list "
            f"({type(exc).__name__})."
        )


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


def _extract_symbol_history(
    download: pd.DataFrame, symbol: str, batch_symbols: tuple[str, ...]
) -> pd.DataFrame:
    """Select one symbol from yfinance's ticker-grouped multi-symbol result."""
    if download is None or download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        for level in range(download.columns.nlevels):
            if symbol in download.columns.get_level_values(level):
                try:
                    return download.xs(symbol, axis=1, level=level, drop_level=True)
                except KeyError:
                    continue
        return pd.DataFrame()
    return download if len(batch_symbols) == 1 else pd.DataFrame()


@_ttl_cache(seconds=3600, maxsize=32)
def fetch_market_data(
    tickers: tuple[str, ...], benchmark: str, period: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, float], dict[str, str]]:
    """Fetch adjusted closes and spot prices; cache results for one hour.

    Symbols are downloaded in small batches so a broad constituent universe
    does not require one request per ticker. A failed or invalid symbol does not
    prevent other symbols in its batch from being used. Historical series are
    returned with their own available dates. Consumers align return series by
    date after calculating each symbol's own close-to-close returns.
    """
    requested = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    symbols = list(dict.fromkeys([*requested, benchmark.strip().upper()]))
    raw_prices: dict[str, pd.Series] = {}
    spots: dict[str, float] = {}
    errors: dict[str, str] = {}

    batch_size = 40
    for start in range(0, len(symbols), batch_size):
        batch_symbols = tuple(symbols[start : start + batch_size])
        try:
            download = yf.download(
                tickers=list(batch_symbols),
                period=period,
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=20,
            )
        except Exception as exc:  # yfinance errors vary by provider and ticker
            for symbol in batch_symbols:
                errors[symbol] = f"Data could not be loaded ({type(exc).__name__})."
            continue

        for symbol in batch_symbols:
            history = _extract_symbol_history(download, symbol, batch_symbols)
            prices = _clean_history(history)
            if prices.empty:
                for alternate in YFINANCE_TICKER_FALLBACKS.get(symbol, ()):
                    try:
                        alternate_download = yf.download(
                            tickers=[alternate],
                            period=period,
                            auto_adjust=False,
                            actions=False,
                            group_by="ticker",
                            threads=False,
                            progress=False,
                            timeout=20,
                        )
                        alternate_history = _extract_symbol_history(
                            alternate_download, alternate, (alternate,)
                        )
                        prices = _clean_history(alternate_history)
                        if not prices.empty:
                            break
                    except Exception:
                        continue
            if prices.empty:
                errors[symbol] = "No historical prices returned. Check the ticker or lookback period."
                continue
            raw_prices[symbol] = prices
            # Use the latest available yfinance adjusted close as the price
            # reference, avoiding a separate quote request for every symbol.
            spots[symbol] = float(prices.iloc[-1])

    benchmark_symbol = benchmark.strip().upper()
    benchmark_prices = raw_prices.get(benchmark_symbol, pd.Series(dtype=float, name="Benchmark"))
    if benchmark_prices.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([])), benchmark_prices, spots, errors

    available: dict[str, pd.Series] = {}
    for symbol in requested:
        series = raw_prices.get(symbol)
        if series is None:
            continue
        pair = pd.concat([series.rename(symbol), benchmark_prices.rename("Benchmark")], axis=1).dropna()
        if pair.empty:
            errors[symbol] = "No overlapping trading dates with the benchmark."
            continue
        available[symbol] = series

    prices_frame = pd.DataFrame(available).sort_index() if available else pd.DataFrame(index=benchmark_prices.index)
    return prices_frame, benchmark_prices, spots, errors

