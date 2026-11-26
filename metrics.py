"""Risk, return-estimation and walk-forward calculations for the dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


TRADING_DAYS = 252
MONTH_TRADING_DAYS = 21
EWMA_DECAY = 0.94
EXCESS_RETURN_PRIOR_SD = 0.10


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


def _ewma_annual_volatility(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return float("nan")
    variance = clean.ewm(alpha=1.0 - EWMA_DECAY, adjust=False).var(bias=False).iloc[-1]
    if not np.isfinite(variance) or variance < 0:
        variance = clean.var(ddof=1)
    return float(np.sqrt(variance * TRADING_DAYS))


def _rolling_compound_returns(returns: pd.Series, window: int) -> pd.Series:
    clean = returns.dropna().astype(float)
    if len(clean) < window:
        return pd.Series(dtype=float)
    return clean.add(1.0).rolling(window).apply(np.prod, raw=True).sub(1.0).dropna()


def _historical_tail_losses(
    returns: pd.Series, window: int = MONTH_TRADING_DAYS, confidence: float = 0.95
) -> dict[str, float]:
    """Return positive historical loss magnitudes over overlapping holding windows."""
    horizon_returns = _rolling_compound_returns(returns, window)
    if horizon_returns.empty:
        return {"var": float("nan"), "cvar": float("nan"), "worst": float("nan")}
    cutoff = float(horizon_returns.quantile(1.0 - confidence))
    tail = horizon_returns[horizon_returns <= cutoff]
    return {
        "var": max(0.0, -cutoff),
        "cvar": max(0.0, -float(tail.mean())) if not tail.empty else max(0.0, -cutoff),
        "worst": max(0.0, -float(horizon_returns.min())),
    }


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
    """Calculate historical metrics and a shrinkage-based return estimate.

    Daily returns are calculated per series before aligning dates, so a missing
    observation in one series does not turn a multi-session move in another into
    a one-session return. Tail losses use overlapping 21-session historical
    returns. The return estimate shrinks measured excess return toward zero.
    """
    stock_prices = _clean_prices(prices)
    market_prices = _clean_prices(benchmark_prices)
    stock_returns = stock_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    paired_returns = pd.concat(
        [stock_returns.rename("stock"), market_returns.rename("market")], axis=1
    ).dropna()
    if stock_returns.empty:
        return {}

    rf_annual = float(annual_risk_free_rate)
    daily_rf = annual_to_daily_rate(rf_annual)
    sample_volatility = (
        float(stock_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(stock_returns) > 1
        else float("nan")
    )
    volatility = _ewma_annual_volatility(stock_returns)

    # A normal prior centered on zero expected excess return limits extreme
    # historical-mean estimates. Prior SD is disclosed in the app's methodology.
    observed_excess = float((stock_returns.mean() - daily_rf) * TRADING_DAYS)
    if np.isfinite(sample_volatility) and sample_volatility > 0:
        annual_mean_se = sample_volatility * np.sqrt(TRADING_DAYS / len(stock_returns))
        prior_var = EXCESS_RETURN_PRIOR_SD**2
        posterior_weight = prior_var / (prior_var + annual_mean_se**2)
    else:
        posterior_weight = 0.0
    annual_expected = float(rf_annual + posterior_weight * observed_excess)

    initial = float(stock_prices.iloc[0])
    final = float(stock_prices.iloc[-1])
    cagr = float((final / initial) ** (TRADING_DAYS / len(stock_returns)) - 1.0)

    downside = np.minimum(stock_returns.to_numpy(dtype=float) - daily_rf, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)) * TRADING_DAYS))

    running_peak = stock_prices.cummax()
    drawdowns = (stock_prices / running_peak) - 1.0
    max_drawdown = float(max(0.0, -drawdowns.min()))

    capm, _ = _capm_fit(paired_returns["stock"], paired_returns["market"], daily_rf)
    beta = capm["beta"]
    alpha_annual = (
        capm["alpha_daily"] * TRADING_DAYS
        if np.isfinite(capm["alpha_daily"])
        else float("nan")
    )
    excess_return = annual_expected - rf_annual
    tail = _historical_tail_losses(stock_returns)

    spot = (
        float(current_price)
        if current_price is not None and np.isfinite(current_price) and current_price > 0
        else final
    )
    return {
        "observations": int(len(stock_returns)),
        "current_price": spot,
        "cagr": cagr,
        "expected_return": annual_expected,
        "return_signal_weight": float(posterior_weight),
        "volatility": volatility,
        "historical_volatility": sample_volatility,
        "beta": beta,
        "downside_deviation": downside_deviation,
        "max_drawdown": max_drawdown,
        "var95_1m": tail["var"],
        "cvar95_1m": tail["cvar"],
        "worst_1m_loss": tail["worst"],
        "sharpe": _safe_ratio(excess_return, volatility),
        "sortino": _safe_ratio(excess_return, downside_deviation),
        "treynor": _safe_ratio(excess_return, beta),
        "jensen_alpha": alpha_annual,
        "r_squared": capm["r_squared"],
        "daily_risk_free_rate": daily_rf,
        "price_series": stock_prices,
        "benchmark_series": market_prices,
    }


def volatility_target_backtest(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    annual_risk_free_rate: float,
    target_annual_volatility: float = 0.15,
    transaction_cost_bps: float = 10.0,
    lookback: int = 63,
) -> dict[str, Any]:
    """Simulate a long-only, unlevered, lagged-volatility exposure rule.

    Exposure for each session is set using only returns available before that
    session. Uninvested exposure earns the supplied risk-free rate. Estimated
    transaction costs apply to exposure changes and an assumed final exit.
    """
    stock_returns = _clean_prices(prices).pct_change().rename("stock").dropna()
    market_returns = _clean_prices(benchmark_prices).pct_change().rename("benchmark").dropna()
    paired = pd.concat([stock_returns, market_returns], axis=1).dropna()
    empty = {"summary": pd.DataFrame(), "wealth": pd.DataFrame(), "exposure": pd.Series(dtype=float)}
    if len(paired) <= lookback + 1:
        return empty

    rolling_vol = paired["stock"].rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(TRADING_DAYS)
    target = max(0.01, float(target_annual_volatility))
    exposure = (target / rolling_vol).clip(lower=0.0, upper=1.0)
    exposure = exposure.mask(rolling_vol.eq(0.0), 1.0).shift(1).dropna()
    simulation = paired.loc[exposure.index].copy()
    simulation["exposure"] = exposure
    if simulation.empty:
        return empty

    daily_rf = annual_to_daily_rate(annual_risk_free_rate)
    cost_rate = max(0.0, float(transaction_cost_bps)) / 10000.0
    turnover = simulation["exposure"].diff().abs()
    turnover.iloc[0] = abs(float(simulation["exposure"].iloc[0]))
    turnover.iloc[-1] += abs(float(simulation["exposure"].iloc[-1]))  # assumed close-out

    targeted = (
        simulation["exposure"] * simulation["stock"]
        + (1.0 - simulation["exposure"]) * daily_rf
        - turnover * cost_rate
    )
    stock_buy_hold = simulation["stock"].copy()
    benchmark_buy_hold = simulation["benchmark"].copy()
    stock_buy_hold.iloc[0] -= cost_rate
    benchmark_buy_hold.iloc[0] -= cost_rate
    stock_buy_hold.iloc[-1] -= cost_rate
    benchmark_buy_hold.iloc[-1] -= cost_rate

    returns = pd.DataFrame(
        {
            "Volatility-targeted (after costs)": targeted,
            "Stock buy-and-hold (after costs)": stock_buy_hold,
            "Benchmark buy-and-hold (after costs)": benchmark_buy_hold,
        }
    )
    wealth = (1.0 + returns).cumprod()
    summary_rows: list[dict[str, float | str]] = []
    years = len(returns) / TRADING_DAYS
    for name in returns:
        series = returns[name]
        total_return = float(wealth[name].iloc[-1] - 1.0)
        cagr = (
            float(wealth[name].iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth[name].iloc[-1] > 0
            else float("nan")
        )
        annual_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(series) > 1 else float("nan")
        drawdown = wealth[name] / wealth[name].cummax() - 1.0
        one_month = _rolling_compound_returns(series, MONTH_TRADING_DAYS)
        summary_rows.append(
            {
                "Strategy": name,
                "Total Return": total_return,
                "CAGR": cagr,
                "Annual Volatility": annual_vol,
                "Max Drawdown": float(max(0.0, -drawdown.min())),
                "Worst 1M Loss": max(0.0, -float(one_month.min())) if not one_month.empty else float("nan"),
                "Sharpe": _safe_ratio(float((series.mean() - daily_rf) * TRADING_DAYS), annual_vol),
            }
        )
    return {
        "summary": pd.DataFrame(summary_rows),
        "wealth": wealth,
        "exposure": simulation["exposure"],
        "estimated_turnover": float(turnover.sum()),
        "estimated_cost_drag": float((turnover * cost_rate).sum()),
    }


def capm_regression_data(
    prices: pd.Series, benchmark_prices: pd.Series, annual_risk_free_rate: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return paired excess-return observations and fitted CAPM coefficients."""
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

