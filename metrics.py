"""Pure financial calculations used by the stock screener dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm


TRADING_DAYS = 252


def annual_to_daily_rate(annual_rate: float) -> float:
    """Convert an effective annual rate to an effective daily rate."""
    rate = float(annual_rate)
    if not np.isfinite(rate) or rate <= -1:
        return 0.0
    return float(np.expm1(np.log1p(rate) / TRADING_DAYS))


def _clean_prices(prices: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    cleaned = cleaned[(cleaned > 0) & np.isfinite(cleaned)]
    return cleaned[~cleaned.index.duplicated(keep="last")].sort_index()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Return positive VaR loss magnitude from a normal return model."""
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    mean = float(clean.mean())
    volatility = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    if volatility <= 1e-15:
        return max(0.0, -mean)
    lower_tail_return = float(norm.ppf(1.0 - confidence, loc=mean, scale=volatility))
    return max(0.0, -lower_tail_return)


def _capm_fit(
    stock_returns: pd.Series, benchmark_returns: pd.Series, daily_rf: float
) -> tuple[dict[str, float], pd.DataFrame]:
    paired = pd.concat(
        [stock_returns.rename("stock"), benchmark_returns.rename("market")], axis=1
    ).dropna()
    empty = {"beta": float("nan"), "alpha_daily": float("nan"), "r_squared": float("nan")}
    if len(paired) < 3:
        return empty, pd.DataFrame(columns=["market_excess", "stock_excess", "fitted"])

    x = paired["market"] - daily_rf
    y = paired["stock"] - daily_rf
    if not np.isfinite(x).all() or not np.isfinite(y).all() or float(x.var()) < 1e-16:
        return empty, pd.DataFrame(columns=["market_excess", "stock_excess", "fitted"])

    try:
        design = sm.add_constant(x, has_constant="add")
        result = sm.OLS(y, design, missing="drop").fit()
        alpha = float(result.params.iloc[0])
        beta = float(result.params.iloc[1])
        fit_data = pd.DataFrame(
            {"market_excess": x, "stock_excess": y, "fitted": result.predict(design)}
        )
        return {"beta": beta, "alpha_daily": alpha, "r_squared": float(result.rsquared)}, fit_data
    except (ValueError, np.linalg.LinAlgError, IndexError):
        return empty, pd.DataFrame(columns=["market_excess", "stock_excess", "fitted"])


def calculate_stock_metrics(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    annual_risk_free_rate: float,
    current_price: float | None = None,
) -> dict[str, Any]:
    """Calculate return, risk, CAPM and one-year estimate metrics for a stock.

    VaR and drawdown fields are positive loss magnitudes. Annualized returns and
    volatilities use 252 trading days. Returns are based on adjusted closes.
    """
    stock_prices = _clean_prices(prices)
    market_prices = _clean_prices(benchmark_prices)
    aligned_prices = pd.concat(
        [stock_prices.rename("stock"), market_prices.rename("market")], axis=1
    ).dropna()
    if aligned_prices.empty:
        return {}

    stock_prices = aligned_prices["stock"]
    market_prices = aligned_prices["market"]
    simple_returns = stock_prices.pct_change().dropna()
    log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
    market_returns = market_prices.pct_change().dropna()
    if simple_returns.empty:
        return {}

    rf_annual = float(annual_risk_free_rate)
    daily_rf = annual_to_daily_rate(rf_annual)
    annual_expected = float(simple_returns.mean() * TRADING_DAYS)
    volatility = float(simple_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(simple_returns) > 1 else 0.0
    initial = float(stock_prices.iloc[0])
    final = float(stock_prices.iloc[-1])
    cagr = float((final / initial) ** (TRADING_DAYS / len(simple_returns)) - 1.0)

    below_target = simple_returns[simple_returns < daily_rf] - daily_rf
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(below_target))) * np.sqrt(TRADING_DAYS))
        if not below_target.empty
        else 0.0
    )

    running_peak = stock_prices.cummax()
    drawdowns = (stock_prices / running_peak) - 1.0
    max_drawdown = float(max(0.0, -drawdowns.min()))

    capm, _ = _capm_fit(simple_returns, market_returns, daily_rf)
    beta = capm["beta"]
    alpha_annual = capm["alpha_daily"] * TRADING_DAYS if np.isfinite(capm["alpha_daily"]) else float("nan")
    excess_return = annual_expected - rf_annual

    daily_var95 = parametric_var(simple_returns, 0.95)
    daily_var99 = parametric_var(simple_returns, 0.99)
    if volatility <= 1e-15:
        annual_var95 = annual_var99 = max(0.0, -annual_expected)
    else:
        annual_var95 = max(0.0, -float(norm.ppf(0.05, loc=annual_expected, scale=volatility)))
        annual_var99 = max(0.0, -float(norm.ppf(0.01, loc=annual_expected, scale=volatility)))

    spot = float(current_price) if current_price is not None and np.isfinite(current_price) and current_price > 0 else final
    # The requested projection formula uses annual arithmetic return and volatility.
    log_center = np.log(spot) + annual_expected - 0.5 * volatility**2
    lower_price = float(np.exp(np.clip(log_center - volatility, -700, 700)))
    upper_price = float(np.exp(np.clip(log_center + volatility, -700, 700)))

    return {
        "observations": int(len(simple_returns)),
        "current_price": spot,
        "cagr": cagr,
        "expected_return": annual_expected,
        "volatility": volatility,
        "beta": beta,
        "downside_deviation": downside_deviation,
        "max_drawdown": max_drawdown,
        "daily_var95": daily_var95,
        "daily_var99": daily_var99,
        "var95": annual_var95,
        "var99": annual_var99,
        "sharpe": _safe_ratio(excess_return, volatility),
        "sortino": _safe_ratio(excess_return, downside_deviation),
        "treynor": _safe_ratio(excess_return, beta),
        "jensen_alpha": alpha_annual,
        "r_squared": capm["r_squared"],
        "tentative_price_low": lower_price,
        "tentative_price_high": upper_price,
        "tentative_var95": annual_var95,
        "daily_risk_free_rate": daily_rf,
        "log_return_mean_daily": float(log_returns.mean()) if not log_returns.empty else float("nan"),
        "price_series": stock_prices,
        "benchmark_series": market_prices,
    }


def capm_regression_data(
    prices: pd.Series, benchmark_prices: pd.Series, annual_risk_free_rate: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return excess-return observations and fitted CAPM coefficients for charts."""
    stock = _clean_prices(prices).pct_change().dropna()
    market = _clean_prices(benchmark_prices).pct_change().dropna()
    fit, fit_data = _capm_fit(stock, market, annual_to_daily_rate(annual_risk_free_rate))
    return fit_data, {
        "beta": fit["beta"],
        "alpha_annual": fit["alpha_daily"] * TRADING_DAYS
        if np.isfinite(fit["alpha_daily"])
        else float("nan"),
        "r_squared": fit["r_squared"],
    }
