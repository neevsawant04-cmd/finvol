"""Streamlit dashboard for stock risk, return and CAPM screening."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import MARKET_PRESETS, fetch_market_data
from metrics import calculate_stock_metrics, capm_regression_data


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
st.caption("Screen NSE and NASDAQ stocks using historical returns, risk measures and CAPM estimates.")

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

requested_tickers = list(preset["tickers"])
for item in custom_input.split(","):
    ticker = item.strip().upper()
    if ticker and ticker not in requested_tickers:
        requested_tickers.append(ticker)

with st.spinner("Loading adjusted prices and benchmark data…"):
    prices, benchmark_prices, spot_prices, load_errors = fetch_market_data(
        tuple(requested_tickers), str(preset["benchmark"]), period
    )

if load_errors:
    unavailable = ", ".join(load_errors)
    st.warning(f"Some symbols could not be loaded: {unavailable}")
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
    return_choice = st.selectbox("Return filter metric", ["Expected Return", "CAGR"])
    return_column = "expected_return" if return_choice == "Expected Return" else "cagr"
    return_range = _range_slider(
        f"{return_choice} range (%)", universe[return_column] * 100, "return_filter", "%.1f%%"
    )
    max_beta = _max_slider("Maximum beta", universe["beta"], "beta_filter", "%.2f")
    max_drawdown = _max_slider(
        "Maximum drawdown (%)", universe["max_drawdown"] * 100, "drawdown_filter", "%.1f%%"
    )
    max_var95 = _max_slider("Maximum annual VaR 95% (%)", universe["var95"] * 100, "var_filter", "%.1f%%")
    volatility_range = _range_slider(
        "Annual volatility range (%)", universe["volatility"] * 100, "volatility_filter", "%.1f%%"
    )

matches = universe[
    universe["current_price"].between(*price_range)
    & (universe[return_column] * 100).between(*return_range)
    & universe["beta"].le(max_beta)
    & (universe["max_drawdown"] * 100).le(max_drawdown)
    & (universe["var95"] * 100).le(max_var95)
    & (universe["volatility"] * 100).between(*volatility_range)
].copy()

st.subheader("Portfolio screen")
st.caption(
    f"Benchmark: {preset['benchmark']} · Lookback: {period} · Risk-free rate: {annual_rf:.2%} · "
    f"Prices and returns use adjusted daily closes; spot prices use the latest available quote."
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Matching stocks", f"{len(matches)} / {len(universe)}")
k2.metric("Average expected return", _percent(float(matches["expected_return"].mean())) if not matches.empty else "—")
k3.metric("Average volatility", _percent(float(matches["volatility"].mean())) if not matches.empty else "—")
best_sharpe = matches["sharpe"].max() if not matches.empty else np.nan
k4.metric("Best Sharpe ratio", f"{best_sharpe:.2f}" if np.isfinite(best_sharpe) else "—")

if matches.empty:
    st.info("No stocks match the current filter settings. Widen one or more sidebar ranges to see results.")
    st.caption("Tentative returns and risks are historical statistical estimates, not financial advice.")
    st.stop()

st.subheader("Interactive screener")
display = matches[
    [
        "current_price",
        "cagr",
        "expected_return",
        "tentative_price_low",
        "tentative_price_high",
        "volatility",
        "beta",
        "max_drawdown",
        "var95",
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
        "expected_return": "Expected Return",
        "tentative_price_low": "Tentative 1Y Low",
        "tentative_price_high": "Tentative 1Y High",
        "volatility": "Volatility",
        "beta": "Beta",
        "max_drawdown": "Max Drawdown",
        "var95": "VaR 95% (annual)",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "treynor": "Treynor",
        "jensen_alpha": "Jensen's Alpha",
        "r_squared": "CAPM R²",
    }
)
display.insert(
    5,
    "Tentative 1Y Range",
    display["Tentative 1Y Low"].map(lambda value: _money(value, currency))
    + " – "
    + display["Tentative 1Y High"].map(lambda value: _money(value, currency)),
)
display = display.drop(columns=["Tentative 1Y Low", "Tentative 1Y High"])
percent_columns = ["CAGR", "Expected Return", "Volatility", "Max Drawdown", "VaR 95% (annual)", "Jensen's Alpha"]
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
scatter_data["Annualized Expected Return (%)"] = scatter_data["expected_return"] * 100
scatter_data["Current Price"] = scatter_data["current_price"]
fig = px.scatter(
    scatter_data,
    x="Annualized Volatility (%)",
    y="Annualized Expected Return (%)",
    size="Current Price",
    color="sharpe",
    hover_name="ticker",
    hover_data={
        "Current Price": ":,.2f",
        "cagr": ":.2%",
        "beta": ":.2f",
        "max_drawdown": ":.2%",
        "var95": ":.2%",
        "sharpe": ":.2f",
        "sortino": ":.2f",
        "treynor": ":.2f",
        "jensen_alpha": ":.2%",
        "r_squared": ":.2f",
    },
    color_continuous_scale="RdYlGn",
    size_max=48,
    labels={"sharpe": "Sharpe Ratio"},
    title="Annualized volatility and expected return",
)
fig.update_layout(xaxis_title="Annualized volatility (%)", yaxis_title="Annualized expected return (%)", coloraxis_colorbar_title="Sharpe")
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
    "Tentative 1-year price ranges and VaR are statistical estimates based on historical data and the selected lookback period. "
    "They are not forecasts or financial advice."
)
