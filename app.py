"""Streamlit dashboard for stock risk, return and CAPM screening."""

from __future__ import annotations

from typing import Iterable
import importlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_loader as _data_loader
from metrics import (
    calculate_stock_metrics,
    capm_regression_data,
    volatility_target_backtest,
)


# Streamlit may rerun app.py while retaining an older imported data_loader
# module. Reload it if the current source has the NIFTY constituent loader but
# the in-memory module does not yet expose it.
if not hasattr(_data_loader, "load_nifty500_tickers"):
    _data_loader = importlib.reload(_data_loader)

MARKET_PRESETS = _data_loader.MARKET_PRESETS
fetch_market_data = _data_loader.fetch_market_data


def load_nifty500_tickers() -> tuple[tuple[str, ...], str]:
    loader = getattr(_data_loader, "load_nifty500_tickers", None)
    if loader is not None:
        return loader()
    fallback = tuple(MARKET_PRESETS["Indian Market (NSE)"]["tickers"])
    return fallback, (
        "This data_loader.py is an older version; restart Streamlit to load the current NIFTY 500 support."
    )


st.set_page_config(page_title="Stock Risk & Return Calculator", page_icon="📈", layout="wide")


def _finite_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
            if np.isfinite(number):
                result.append(number)
        except (TypeError, ValueError):
            continue
    return result


def _range_slider(label: str, values: Iterable[object], key: str, format_string: str | None = None) -> tuple[float, float]:
    finite = _finite_values(values)
    if finite:
        lower, upper = min(finite), max(finite)
    else:
        lower, upper = 0.0, 1.0
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 0.01)
        lower, upper = lower - pad, upper + pad
    kwargs = {"format": format_string} if format_string else {}
    return st.slider(
        label,
        min_value=float(lower),
        max_value=float(upper),
        value=(float(lower), float(upper)),
        key=key,
        **kwargs,
    )


def _max_slider(label: str, values: Iterable[object], key: str, format_string: str | None = None) -> float:
    finite = _finite_values(values)
    if finite:
        lower, upper = min(0.0, min(finite)), max(0.0, max(finite))
    else:
        lower, upper = 0.0, 1.0
    if np.isclose(lower, upper):
        upper = lower + max(abs(lower) * 0.05, 0.01)
    kwargs = {"format": format_string} if format_string else {}
    return float(
        st.slider(
            label,
            min_value=float(lower),
            max_value=float(upper),
            value=float(upper),
            key=key,
            **kwargs,
        )
    )


def _money(value: float, currency: str) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{currency}{value:,.2f}"


def _percent(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}%}"


st.title("Stock Risk & Return Calculator")
st.caption("Screen NSE and NASDAQ stocks with conservative return estimates, historical tail risk and a cost-aware risk-control simulation.")

with st.sidebar:
    st.header("Market & lookback")
    market_name = st.selectbox("Market", list(MARKET_PRESETS.keys()))
    preset = MARKET_PRESETS[market_name]
    period = st.selectbox("Lookback period", ["1y", "3y", "5y"], index=1)
    annual_rf = st.number_input(
        "Annual risk-free rate (%)",
        min_value=-10.0,
        max_value=30.0,
        value=float(preset["risk_free_rate"] * 100),
        step=0.05,
        format="%.2f",
    ) / 100.0
    custom_input = st.text_input(
        "Add custom tickers (comma-separated)",
        placeholder="e.g. WIPRO.NS, HCLTECH.NS or AMD",
    )

universe_warning = ""
if market_name == "Indian Market (NSE)":
    default_tickers, universe_warning = load_nifty500_tickers()
else:
    default_tickers = tuple(preset["tickers"])

requested_tickers = list(default_tickers)
for item in custom_input.split(","):
    ticker = item.strip().upper()
    if ticker and ticker not in requested_tickers:
        requested_tickers.append(ticker)
st.sidebar.caption(f"Default universe: {len(default_tickers)} stocks, plus any custom tickers.")
if universe_warning:
    st.sidebar.warning(universe_warning)

with st.spinner("Loading adjusted prices and benchmark data…"):
    prices, benchmark_prices, spot_prices, load_errors = fetch_market_data(
        tuple(requested_tickers), str(preset["benchmark"]), period
    )

if universe_warning:
    st.warning(universe_warning)
if load_errors:
    error_symbols = list(load_errors)
    preview = ", ".join(error_symbols[:12])
    remainder = f" and {len(error_symbols) - 12} more" if len(error_symbols) > 12 else ""
    st.warning(f"{len(error_symbols)} symbols could not be loaded: {preview}{remainder}")
    with st.expander("Data loading details"):
        for symbol, message in load_errors.items():
            st.write(f"**{symbol}:** {message}")

if benchmark_prices.empty:
    st.error(f"No benchmark history was available for {preset['benchmark']}. Check the connection and try again.")
    st.caption("The dashboard uses Yahoo Finance through yfinance. Historical data availability can vary by symbol and region.")
    st.stop()

rows: list[dict[str, object]] = []
for symbol in prices.columns:
    stock_prices = prices[symbol].dropna()
    calculated = calculate_stock_metrics(
        stock_prices,
        benchmark_prices,
        annual_rf,
        current_price=spot_prices.get(symbol),
    )
    if not calculated:
        continue
    row = {key: value for key, value in calculated.items() if not isinstance(value, (pd.Series, pd.DataFrame))}
    row["ticker"] = symbol
    rows.append(row)

if not rows:
    st.info("No valid stock and benchmark pairs are available for this market and period. Try another lookback or check ticker symbols.")
    st.caption("Tentative returns and risks are historical statistical estimates, not financial advice.")
    st.stop()

universe = pd.DataFrame(rows).set_index("ticker", drop=False)
currency = str(preset["currency"])

with st.sidebar:
    st.divider()
    st.header("Screener filters")
    price_range = _range_slider(
        f"Current price ({currency})", universe["current_price"], "price_filter", "%.2f"
    )
    return_choice = st.selectbox("Return filter metric", ["Shrunk Return Estimate", "CAGR"])
    return_column = "expected_return" if return_choice == "Shrunk Return Estimate" else "cagr"
    return_range = _range_slider(
        f"{return_choice} range (%)", universe[return_column] * 100, "return_filter", "%.1f%%"
    )
    max_beta = _max_slider("Maximum beta", universe["beta"], "beta_filter", "%.2f")
    max_drawdown = _max_slider(
        "Maximum drawdown (%)", universe["max_drawdown"] * 100, "drawdown_filter", "%.1f%%"
    )
    max_cvar95 = _max_slider(
        "Maximum historical 1M CVaR 95% (%)",
        universe["cvar95_1m"] * 100,
        "cvar_filter",
        "%.1f%%",
    )
    volatility_range = _range_slider(
        "Annual volatility range (%)", universe["volatility"] * 100, "volatility_filter", "%.1f%%"
    )

matches = universe[
    universe["current_price"].between(*price_range)
    & (universe[return_column] * 100).between(*return_range)
    & universe["beta"].le(max_beta)
    & (universe["max_drawdown"] * 100).le(max_drawdown)
    & (universe["cvar95_1m"] * 100).le(max_cvar95)
    & (universe["volatility"] * 100).between(*volatility_range)
].copy()

st.subheader("Portfolio screen")
st.caption(
    f"Benchmark: {preset['benchmark']} · Lookback: {period} · Risk-free rate: {annual_rf:.2%} · "
    f"Risk uses adjusted daily closes; price filters use the latest available adjusted close."
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Matching stocks", f"{len(matches)} / {len(universe)}")
k2.metric("Average shrunk return estimate", _percent(float(matches["expected_return"].mean())) if not matches.empty else "—")
k3.metric("Average historical 1M CVaR 95%", _percent(float(matches["cvar95_1m"].mean())) if not matches.empty else "—")
median_volatility = matches["volatility"].median() if not matches.empty else np.nan
k4.metric("Median EWMA volatility", _percent(float(median_volatility)) if np.isfinite(median_volatility) else "—")

with st.expander("Method and assumptions"):
    st.markdown(
        """
        - **Return estimate:** measured excess return is shrunk toward zero using a normal prior with a 10% annual standard deviation. The table also shows the weight placed on the historical signal; it is not a probability of profit.
        - **Volatility:** exponentially weighted daily volatility (decay factor 0.94), annualized over 252 sessions.
        - **Tail loss:** historical 95% VaR and CVaR from overlapping 21-session returns. CVaR is the average loss in the worst 5% of those observed windows. It can understate losses that did not occur in the selected history.
        - **Walk-forward simulation:** uses trailing 63-session volatility to set next-session exposure, caps stock exposure at 100%, credits uninvested cash at the selected risk-free rate, and applies the transaction cost you choose.

        These are historical estimates and simulations, not forecasts or guarantees. The default stock lists represent currently available constituents, so historical comparisons may have survivorship bias. The simulation does not include taxes, market impact, liquidity limits, or intraday execution.
        """
    )

if matches.empty:
    st.info("No stocks match the current filter settings. Widen one or more sidebar ranges to see results.")
    st.caption("Return and risk estimates are historical statistics, not forecasts or guarantees.")
    st.stop()

st.subheader("Interactive screener")
display = matches[
    [
        "current_price",
        "cagr",
        "expected_return",
        "return_signal_weight",
        "volatility",
        "historical_volatility",
        "beta",
        "max_drawdown",
        "var95_1m",
        "cvar95_1m",
        "worst_1m_loss",
        "sharpe",
        "sortino",
        "treynor",
        "jensen_alpha",
        "r_squared",
    ]
].copy()
display.insert(0, "Ticker", display.index)
display = display.rename(
    columns={
        "current_price": "Current Price",
        "cagr": "CAGR",
        "expected_return": "Shrunk Return Estimate",
        "return_signal_weight": "Return Signal Weight",
        "volatility": "EWMA Volatility",
        "historical_volatility": "Historical Volatility",
        "beta": "Beta",
        "max_drawdown": "Max Drawdown",
        "var95_1m": "Historical VaR 95% (1M)",
        "cvar95_1m": "Historical CVaR 95% (1M)",
        "worst_1m_loss": "Worst Historical 1M Loss",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "treynor": "Treynor",
        "jensen_alpha": "Jensen's Alpha",
        "r_squared": "CAPM R²",
    }
)
percent_columns = [
    "CAGR",
    "Shrunk Return Estimate",
    "Return Signal Weight",
    "EWMA Volatility",
    "Historical Volatility",
    "Max Drawdown",
    "Historical VaR 95% (1M)",
    "Historical CVaR 95% (1M)",
    "Worst Historical 1M Loss",
    "Jensen's Alpha",
]
for column in percent_columns:
    display[column] = display[column].map(lambda value: _percent(float(value)))
for column in ["Current Price"]:
    display[column] = display[column].map(lambda value: _money(float(value), currency))
for column in ["Beta", "Sharpe", "Sortino", "Treynor", "CAPM R²"]:
    display[column] = display[column].map(lambda value: f"{float(value):.2f}" if np.isfinite(value) else "—")

st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)
csv_frame = matches.reset_index(drop=True).rename(columns={"ticker": "Ticker"})
csv_bytes = csv_frame.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name=f"stock_risk_return_{market_name.lower().replace(' ', '_')}_{period}.csv",
    mime="text/csv",
)

st.subheader("Risk versus return")
scatter_data = matches.copy()
scatter_data["Annualized Volatility (%)"] = scatter_data["volatility"] * 100
scatter_data["Shrunk Return Estimate (%)"] = scatter_data["expected_return"] * 100
scatter_data["Current Price"] = scatter_data["current_price"]
fig = px.scatter(
    scatter_data,
    x="Annualized Volatility (%)",
    y="Shrunk Return Estimate (%)",
    size="Current Price",
    color="sharpe",
    hover_name="ticker",
    hover_data={
        "Current Price": ":,.2f",
        "cagr": ":.2%",
        "beta": ":.2f",
        "max_drawdown": ":.2%",
        "cvar95_1m": ":.2%",
        "sharpe": ":.2f",
        "sortino": ":.2f",
        "treynor": ":.2f",
        "jensen_alpha": ":.2%",
        "r_squared": ":.2f",
    },
    color_continuous_scale="RdYlGn",
    size_max=48,
    labels={"sharpe": "Sharpe Ratio"},
    title="EWMA volatility and shrunk return estimate",
)
fig.update_layout(xaxis_title="Annualized EWMA volatility (%)", yaxis_title="Shrunk return estimate (%)", coloraxis_colorbar_title="Estimated Sharpe")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Deep-dive stock inspector")
selected = st.selectbox("Stock", list(matches.index), format_func=lambda symbol: str(symbol))
selected_prices = prices[selected].dropna()
pair = pd.concat(
    [selected_prices.rename("Stock"), benchmark_prices.rename("Benchmark")], axis=1
).dropna()

left, right = st.columns(2)
with left:
    relative = pd.DataFrame(
        {
            "Stock": pair["Stock"] / pair["Stock"].iloc[0] - 1.0,
            "Benchmark": pair["Benchmark"] / pair["Benchmark"].iloc[0] - 1.0,
        }
    )
    relative_fig = px.line(
        relative.rename_axis("Date").reset_index(),
        x="Date",
        y=["Stock", "Benchmark"],
        title="Cumulative return versus benchmark",
    )
    relative_fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative return", yaxis_tickformat=".0%", legend_title="Series")
    st.plotly_chart(relative_fig, use_container_width=True)

with right:
    stock_drawdown = pair["Stock"] / pair["Stock"].cummax() - 1.0
    benchmark_drawdown = pair["Benchmark"] / pair["Benchmark"].cummax() - 1.0
    underwater = pd.DataFrame({"Stock": stock_drawdown, "Benchmark": benchmark_drawdown})
    drawdown_fig = px.area(
        underwater.rename_axis("Date").reset_index(),
        x="Date",
        y=["Stock", "Benchmark"],
        title="Historical drawdown (underwater chart)",
    )
    drawdown_fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%", legend_title="Series")
    st.plotly_chart(drawdown_fig, use_container_width=True)

st.subheader("Walk-forward risk-control simulation")
st.caption(
    "A historical paper simulation for the selected stock. Each session's exposure uses only earlier returns; "
    "the rule is not optimized to maximize historical performance."
)
setting_left, setting_right = st.columns(2)
with setting_left:
    target_volatility = st.slider(
        "Target annual volatility",
        min_value=5,
        max_value=40,
        value=15,
        step=1,
        format="%d%%",
    ) / 100.0
with setting_right:
    transaction_cost_bps = st.slider(
        "Estimated transaction cost per side (basis points)",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
    )

simulation = volatility_target_backtest(
    selected_prices,
    benchmark_prices,
    annual_rf,
    target_annual_volatility=target_volatility,
    transaction_cost_bps=transaction_cost_bps,
)
if simulation["summary"].empty:
    st.info("There is not enough aligned history for the 63-session warm-up and walk-forward comparison.")
else:
    summary_display = simulation["summary"].copy()
    for column in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Worst 1M Loss"]:
        summary_display[column] = summary_display[column].map(lambda value: _percent(float(value)))
    summary_display["Sharpe"] = summary_display["Sharpe"].map(
        lambda value: f"{float(value):.2f}" if np.isfinite(value) else "—"
    )
    st.dataframe(summary_display, use_container_width=True, hide_index=True)
    st.caption(
        f"Estimated total turnover: {simulation['estimated_turnover']:.2f}× · "
        f"Estimated transaction-cost drag: {_percent(simulation['estimated_cost_drag'])}. "
        "The rule can lag a rising market and still lose money; lower simulated volatility does not guarantee a smaller future loss."
    )
    wealth_frame = simulation["wealth"].rename_axis("Date").reset_index()
    wealth_fig = px.line(
        wealth_frame,
        x="Date",
        y=list(simulation["wealth"].columns),
        title="After-cost cumulative value (starting at 1.0)",
    )
    wealth_fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio value", legend_title="Simulation")
    st.plotly_chart(wealth_fig, use_container_width=True)
    exposure_fig = px.line(
        simulation["exposure"].rename("Stock exposure").rename_axis("Date").reset_index(),
        x="Date",
        y="Stock exposure",
        title="Lagged, unlevered stock exposure",
    )
    exposure_fig.update_layout(xaxis_title="Date", yaxis_title="Exposure", yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(exposure_fig, use_container_width=True)

st.markdown("**CAPM regression: stock excess return versus benchmark excess return**")
fit_data, capm_stats = capm_regression_data(pair["Stock"], pair["Benchmark"], annual_rf)
if fit_data.empty:
    st.info("There are not enough non-constant paired return observations to fit the CAPM regression.")
else:
    reg_fig = go.Figure()
    reg_fig.add_trace(
        go.Scatter(
            x=fit_data["market_excess"],
            y=fit_data["stock_excess"],
            mode="markers",
            name="Daily observations",
            marker={"size": 6, "opacity": 0.55},
        )
    )
    sorted_fit = fit_data.sort_values("market_excess")
    reg_fig.add_trace(
        go.Scatter(
            x=sorted_fit["market_excess"],
            y=sorted_fit["fitted"],
            mode="lines",
            name="OLS fitted line",
            line={"width": 3},
        )
    )
    reg_fig.update_layout(
        title=f"{selected}: β {capm_stats['beta']:.3f} · R² {capm_stats['r_squared']:.3f} · Jensen's α {capm_stats['alpha_annual']:.2%}",
        xaxis_title="Benchmark excess daily return",
        yaxis_title="Stock excess daily return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
    )
    st.plotly_chart(reg_fig, use_container_width=True)

st.caption(
    "Historical risk and return estimates do not predict future outcomes. The walk-forward simulation is illustrative and cannot eliminate losses."
)

