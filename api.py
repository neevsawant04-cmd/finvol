"""FastAPI backend for stock screening, history and risk-control simulations."""

from __future__ import annotations

import os
import re
import secrets
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from data_loader import MARKET_PRESETS, fetch_market_data, load_nifty50_tickers
from metrics import calculate_stock_metrics, volatility_target_backtest


MarketName = Literal["Indian Market (NSE)"]
Period = Literal["1y", "3y", "5y"]
MAX_SCREEN_TICKERS = 50
TICKER_PATTERN = re.compile(r"^[A-Z0-9&.^=_-]{1,32}$")

app = FastAPI(
    title="Stock Risk & Return API",
    version="1.0.0",
    description="Market data, conservative return estimates, historical tail risk, and walk-forward simulations.",
)


class AnalysisRequest(BaseModel):
    market: MarketName
    ticker: str = Field(min_length=1, max_length=32)
    period: Period = "3y"
    annual_risk_free_rate: float | None = Field(default=None, ge=-0.10, le=0.30)


class ScreenRequest(BaseModel):
    market: MarketName
    tickers: list[str] = Field(default_factory=list, max_length=MAX_SCREEN_TICKERS)
    period: Period = "3y"
    annual_risk_free_rate: float | None = Field(default=None, ge=-0.10, le=0.30)


class SimulationRequest(AnalysisRequest):
    target_annual_volatility: float = Field(default=0.15, ge=0.05, le=0.40)
    transaction_cost_bps: float = Field(default=10.0, ge=0.0, le=1000.0)


class AnalysisItem(BaseModel):
    ticker: str
    status: str = "ready"
    observations: int
    data_start: str | None
    data_end: str | None
    metrics: dict[str, float | None]


class ScreenResponse(BaseModel):
    market: MarketName
    benchmark: str
    period: Period
    annual_risk_free_rate: float
    results: list[AnalysisItem]
    load_errors: dict[str, str]
    warnings: list[str]


class PricePoint(BaseModel):
    date: str
    price: float


class HistoryResponse(BaseModel):
    market: MarketName
    ticker: str
    benchmark: str
    period: Period
    stock_prices: list[PricePoint]
    benchmark_prices: list[PricePoint]


class SimulationResponse(BaseModel):
    market: MarketName
    ticker: str
    benchmark: str
    summary: list[dict[str, float | str | None]]
    wealth: list[dict[str, float | str | None]]
    exposure: list[dict[str, float | str]]
    estimated_turnover: float
    estimated_cost_drag: float


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require a shared token when STOCK_API_KEY is configured on the server."""
    expected = os.getenv("STOCK_API_KEY", "").strip()
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")


def _market_settings(market: MarketName) -> dict:
    return MARKET_PRESETS[market]


def _risk_free_rate(market: MarketName, supplied: float | None) -> float:
    return float(_market_settings(market)["risk_free_rate"] if supplied is None else supplied)


def _ticker(value: str) -> str:
    symbol = value.strip().upper()
    if not TICKER_PATTERN.fullmatch(symbol) or not symbol.endswith(".NS"):
        raise HTTPException(status_code=422, detail=f"Only valid NSE symbols ending in .NS are supported: {value!r}")
    return symbol


def _screen_tickers(market: MarketName, requested: list[str]) -> tuple[list[str], str]:
    warning = ""
    if not requested:
        symbols, warning = load_nifty50_tickers()
        requested = list(symbols)
    symbols = list(dict.fromkeys(_ticker(value) for value in requested))
    if len(symbols) > MAX_SCREEN_TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"A screen can contain at most {MAX_SCREEN_TICKERS} NSE tickers.",
        )
    return symbols, warning


def _finite_number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _analysis_item(ticker: str, metrics: dict) -> AnalysisItem:
    series = metrics.get("price_series")
    scalar_metrics = {
        key: _finite_number(value)
        for key, value in metrics.items()
        if not isinstance(value, (pd.Series, pd.DataFrame))
    }
    return AnalysisItem(
        ticker=ticker,
        observations=int(metrics.get("observations", 0)),
        data_start=series.index[0].isoformat() if isinstance(series, pd.Series) and not series.empty else None,
        data_end=series.index[-1].isoformat() if isinstance(series, pd.Series) and not series.empty else None,
        metrics=scalar_metrics,
    )


def _load_ticker(
    market: MarketName, ticker: str, period: Period
) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    settings = _market_settings(market)
    benchmark_symbol = str(settings["benchmark"])
    prices, benchmark, _, errors = fetch_market_data((ticker,), benchmark_symbol, period)
    if benchmark.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No data returned for benchmark {benchmark_symbol}. Try again later.",
        )
    if ticker not in prices.columns or prices[ticker].dropna().empty:
        raise HTTPException(
            status_code=404,
            detail=errors.get(ticker, f"No usable history was returned for {ticker}."),
        )
    return prices[ticker].dropna(), benchmark.dropna(), errors


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness endpoint; intentionally does not require the API key."""
    return {"status": "ok"}


@app.get("/v1/markets", dependencies=[Depends(require_api_key)])
def markets() -> list[dict[str, str | int]]:
    return [
        {
            "market": name,
            "benchmark": str(settings["benchmark"]),
            "currency": str(settings["currency"]),
            "built_in_ticker_count": len(settings["tickers"]),
        }
        for name, settings in MARKET_PRESETS.items()
    ]


@app.post("/v1/screen", response_model=ScreenResponse, dependencies=[Depends(require_api_key)])
def screen(request: ScreenRequest) -> ScreenResponse:
    symbols, universe_warning = _screen_tickers(request.market, request.tickers)
    settings = _market_settings(request.market)
    benchmark_symbol = str(settings["benchmark"])
    risk_free_rate = _risk_free_rate(request.market, request.annual_risk_free_rate)
    prices, benchmark, _, errors = fetch_market_data(tuple(symbols), benchmark_symbol, request.period)
    if benchmark.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No data returned for benchmark {benchmark_symbol}. Try again later.",
        )

    results: list[AnalysisItem] = []
    for symbol in symbols:
        if symbol not in prices.columns:
            results.append(AnalysisItem(
                ticker=symbol, status="unavailable", observations=0,
                data_start=None, data_end=None, metrics={},
            ))
            continue
        calculated = calculate_stock_metrics(
            prices[symbol].dropna(), benchmark, risk_free_rate
        )
        if calculated:
            results.append(_analysis_item(symbol, calculated))
        else:
            results.append(AnalysisItem(
                ticker=symbol, status="unavailable", observations=0,
                data_start=None, data_end=None, metrics={},
            ))
    warnings = [universe_warning] if universe_warning else []
    return ScreenResponse(
        market=request.market,
        benchmark=benchmark_symbol,
        period=request.period,
        annual_risk_free_rate=risk_free_rate,
        results=results,
        load_errors=errors,
        warnings=warnings,
    )


@app.post("/v1/metrics", response_model=AnalysisItem, dependencies=[Depends(require_api_key)])
def metrics(request: AnalysisRequest) -> AnalysisItem:
    ticker = _ticker(request.ticker)
    stock_prices, benchmark, _ = _load_ticker(request.market, ticker, request.period)
    calculated = calculate_stock_metrics(
        stock_prices,
        benchmark,
        _risk_free_rate(request.market, request.annual_risk_free_rate),
    )
    if not calculated:
        raise HTTPException(status_code=422, detail=f"Could not calculate metrics for {ticker}.")
    return _analysis_item(ticker, calculated)


@app.post("/v1/history", response_model=HistoryResponse, dependencies=[Depends(require_api_key)])
def history(request: AnalysisRequest) -> HistoryResponse:
    ticker = _ticker(request.ticker)
    stock_prices, benchmark, _ = _load_ticker(request.market, ticker, request.period)
    return HistoryResponse(
        market=request.market,
        ticker=ticker,
        benchmark=str(_market_settings(request.market)["benchmark"]),
        period=request.period,
        stock_prices=[
            PricePoint(date=pd.Timestamp(date).isoformat(), price=float(value))
            for date, value in stock_prices.items()
        ],
        benchmark_prices=[
            PricePoint(date=pd.Timestamp(date).isoformat(), price=float(value))
            for date, value in benchmark.items()
        ],
    )


@app.post(
    "/v1/simulation",
    response_model=SimulationResponse,
    dependencies=[Depends(require_api_key)],
)
def simulation(request: SimulationRequest) -> SimulationResponse:
    ticker = _ticker(request.ticker)
    stock_prices, benchmark, _ = _load_ticker(request.market, ticker, request.period)
    result = volatility_target_backtest(
        stock_prices,
        benchmark,
        _risk_free_rate(request.market, request.annual_risk_free_rate),
        target_annual_volatility=request.target_annual_volatility,
        transaction_cost_bps=request.transaction_cost_bps,
    )
    if result["summary"].empty:
        raise HTTPException(
            status_code=422,
            detail="There is not enough aligned history for the simulation warm-up.",
        )

    wealth = result["wealth"]
    wealth_rows: list[dict[str, float | str | None]] = []
    for date, row in wealth.iterrows():
        wealth_rows.append(
            {
                "date": pd.Timestamp(date).isoformat(),
                **{name: _finite_number(value) for name, value in row.items()},
            }
        )
    exposure_rows = [
        {"date": pd.Timestamp(date).isoformat(), "exposure": float(value)}
        for date, value in result["exposure"].items()
    ]
    return SimulationResponse(
        market=request.market,
        ticker=ticker,
        benchmark=str(_market_settings(request.market)["benchmark"]),
        summary=[
            {
                key: value if isinstance(value, str) else _finite_number(value)
                for key, value in row.items()
            }
            for row in result["summary"].to_dict(orient="records")
        ],
        wealth=wealth_rows,
        exposure=exposure_rows,
        estimated_turnover=float(result["estimated_turnover"]),
        estimated_cost_drag=float(result["estimated_cost_drag"]),
    )

