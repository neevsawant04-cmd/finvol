"""Dash UI for the NSE-only Stock Risk & Return Calculator."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots

from api import app as server
from data_loader import MARKET_PRESETS, fetch_market_data, load_nifty50_tickers
from metrics import calculate_stock_metrics

PRESET = MARKET_PRESETS["Indian Market (NSE)"]
CURRENCY = str(PRESET["currency"])
TICKERS, UNIVERSE_WARNING = load_nifty50_tickers()


def _layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        margin={"l": 45, "r": 35, "t": 55, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12}, hovermode="x unified",
    )
    return fig


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 14, "color": "#c7cbd1"})
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return _layout(fig)


def _price(value: Any) -> str:
    try:
        number = float(value)
        return f"{CURRENCY}{number:,.2f}" if np.isfinite(number) else "-"
    except (TypeError, ValueError):
        return "-"


def _pct(value: Any) -> str:
    try:
        number = float(value)
        return f"{number:.2%}" if np.isfinite(number) else "-"
    except (TypeError, ValueError):
        return "-"


def _direction(value: float) -> str:
    return "UP" if value > 0.005 else "DOWN" if value < -0.005 else "FLAT"


def _stock_figures(ticker: str, metrics: dict[str, Any]) -> tuple[go.Figure, go.Figure]:
    prices = metrics.get("price_series")
    validation = metrics.get("forecast_validation")
    if not isinstance(prices, pd.Series) or prices.empty:
        message = "No usable adjusted price history returned."
        return (_empty_figure(f"{ticker} - price and forecast", message),
                _empty_figure(f"{ticker} - predicted versus actual", message))

    history = prices.tail(126)
    prediction = float(metrics["forecast_1m_price"])
    low = float(metrics["forecast_1m_lower_price"])
    high = float(metrics["forecast_1m_upper_price"])
    future_date = pd.Timestamp(history.index[-1]) + pd.offsets.BDay(21)
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines",
                                   name="Adjusted close", line={"color": "#64b5f6", "width": 2}))
    price_fig.add_trace(go.Scatter(x=[history.index[-1], future_date],
                                   y=[float(history.iloc[-1]), prediction], mode="lines",
                                   name="Forecast path", line={"color": "#ffca28", "dash": "dash"}))
    price_fig.add_trace(go.Scatter(
        x=[future_date], y=[prediction], mode="markers", name="21-session point estimate",
        marker={"color": "#ffca28", "size": 11, "symbol": "diamond"},
        error_y={"type": "data", "symmetric": False,
                 "array": [max(0, high - prediction)],
                 "arrayminus": [max(0, prediction - low)],
                 "visible": True, "color": "#ffca28", "width": 6},
        hovertemplate="Forecast: %{y:,.2f}<extra></extra>",
    ))
    price_fig.update_layout(title=f"{ticker} - 1-month estimate and 90% historical range",
                            xaxis_title="Trading date", yaxis_title=f"Adjusted price ({CURRENCY})")
    price_fig = _layout(price_fig)

    if not isinstance(validation, pd.DataFrame) or validation.empty:
        error_fig = _empty_figure(
            f"{ticker} - walk-forward forecast deviations",
            "Not enough history for 252 training sessions plus a 21-session holdout.",
        )
    else:
        error_fig = make_subplots(specs=[[{"secondary_y": True}]])
        error_fig.add_trace(go.Scatter(
            x=validation["target_date"], y=validation["actual_return"] * 100,
            mode="lines+markers", name="Actual 21-session return",
            line={"color": "#64b5f6", "width": 2}), secondary_y=False)
        error_fig.add_trace(go.Scatter(
            x=validation["target_date"], y=validation["predicted_return"] * 100,
            mode="lines+markers", name="Predicted 21-session return",
            line={"color": "#ffca28", "dash": "dash", "width": 2}), secondary_y=False)
        error_fig.add_trace(go.Bar(
            x=validation["target_date"], y=validation["deviation"] * 100,
            name="Error (actual - predicted)", marker_color="#ef5350", opacity=0.65),
            secondary_y=True)
        error_fig.update_layout(title=f"{ticker} - monthly walk-forward prediction errors",
                                xaxis_title="Forecast target date", barmode="overlay")
        error_fig.update_yaxes(title_text="Return (%)", ticksuffix="%", secondary_y=False)
        error_fig.update_yaxes(title_text="Actual - predicted (percentage points)",
                               ticksuffix=" pp", secondary_y=True)
        error_fig = _layout(error_fig, 360)
    return price_fig, error_fig


def _screen(period: str) -> tuple[Any, Any, Any]:
    prices, benchmark, spots, errors = fetch_market_data(
        tuple(TICKERS), str(PRESET["benchmark"]), period
    )
    if benchmark.empty:
        return (html.Div("Benchmark history unavailable. Refresh and retry.",
                         className="notice notice-error"), html.Div(), html.Div())
    analyses: dict[str, dict[str, Any]] = {}
    for ticker in TICKERS:
        if ticker in prices.columns and not prices[ticker].dropna().empty:
            analyses[ticker] = calculate_stock_metrics(
                prices[ticker].dropna(), benchmark, float(PRESET["risk_free_rate"]),
                current_price=spots.get(ticker),
            )

    rows: list[dict[str, str]] = []
    cards: list[Any] = []
    for ticker in TICKERS:
        metrics = analyses.get(ticker)
        if not metrics:
            rows.append({
                "Ticker": ticker, "Status": "Unavailable", "Prediction": "-",
                "Predicted 1M Return": "-", "Predicted Price (INR)": "-",
                "90% Price Range (INR)": "-", "Historical MAE (1M)": "-",
                "Validation Windows": "0", "EWMA Volatility": "-",
                "Historical CVaR 95% (1M)": "-",
            })
            note = errors.get(ticker, "No adjusted prices returned.")
            card_children = [html.P(note, className="stock-error")]
            pfig, efig = _stock_figures(ticker, {})
        else:
            ret = float(metrics["forecast_1m_return"])
            low, high = float(metrics["forecast_1m_lower_price"]), float(metrics["forecast_1m_upper_price"])
            rows.append({
                "Ticker": ticker, "Status": "Ready", "Prediction": _direction(ret),
                "Predicted 1M Return": _pct(ret),
                "Predicted Price (INR)": _price(metrics["forecast_1m_price"]),
                "90% Price Range (INR)": f"{_price(low)} to {_price(high)}",
                "Historical MAE (1M)": _pct(metrics["forecast_validation_mae"]),
                "Validation Windows": str(metrics["forecast_validation_count"]),
                "EWMA Volatility": _pct(metrics["volatility"]),
                "Historical CVaR 95% (1M)": _pct(metrics["cvar95_1m"]),
            })
            pfig, efig = _stock_figures(ticker, metrics)
            card_children = [
                html.P(f"Point estimate: {_price(metrics['forecast_1m_price'])} "
                       f"({_pct(ret)}); 90% historical range: {_price(low)} to {_price(high)}",
                       className="forecast-summary"),
                html.P(f"{metrics['forecast_validation_count']} non-overlapping monthly validation windows; "
                       f"mean error (actual - predicted): {_pct(metrics['forecast_validation_bias'])}",
                       className="validation-note"),
            ]
        card_children += [dcc.Graph(figure=pfig, config={"displaylogo": False}),
                          dcc.Graph(figure=efig, config={"displaylogo": False})]
        safe_id = ticker.replace(".", "-").replace("&", "-").replace("^", "")
        cards.append(html.Details(
            [html.Summary(f"{ticker} | {rows[-1]['Prediction']} | {rows[-1]['Predicted 1M Return']}"),
             html.Div(card_children)],
            id=f"stock-{safe_id}", className="stock-card",
        ))

    columns = [{"name": name, "id": name} for name in rows[0]]
    table = dash_table.DataTable(
        data=rows, columns=columns, page_size=50,
        sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#191a1d", "color": "#e8e8e8", "padding": "10px",
                    "textAlign": "left", "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                    "whiteSpace": "normal", "height": "auto", "minWidth": "115px"},
        style_header={"backgroundColor": "#282a2e", "fontWeight": "700",
                      "border": "1px solid #42454a"},
        style_data_conditional=[
            {"if": {"filter_query": '{Status} = "Unavailable"'}, "color": "#ff8a80"},
            {"if": {"filter_query": '{Prediction} = "UP"'}, "color": "#81c784"},
            {"if": {"filter_query": '{Prediction} = "DOWN"'}, "color": "#ff8a80"},
        ],
    )
    summary = html.Div([
        html.Div([html.Strong(f"{len(analyses)}/50"), html.Span(" stocks with data")], className="summary-chip"),
        html.Div([html.Strong(str(len(errors))), html.Span(" loading issues")], className="summary-chip"),
        html.Div([html.Strong("21 sessions"), html.Span(" forecast horizon")], className="summary-chip"),
        html.Div([html.Strong("90%"), html.Span(" historical price range")], className="summary-chip"),
    ], className="summary-row")
    return table, html.Div(cards, className="stock-list"), summary


dash_app = Dash(__name__, server=server, title="NSE Stock Risk & Return Calculator",
                update_title="Calculating forecasts...", suppress_callback_exceptions=True)
dash_app.layout = html.Div([
    html.Header([
        html.Div("NSE | QUANTITATIVE SCREEN", className="eyebrow"),
        html.H1("Stock Risk & Return Calculator"),
        html.P("50 NIFTY 50 shares | conservative one-month estimates | historical forecast error",
               className="subtitle"),
    ], className="hero"),
    html.Section([
        html.Div([
            html.Label("Historical lookback", htmlFor="lookback-period"),
            dcc.Dropdown(id="lookback-period", options=[
                {"label": "1 year", "value": "1y"},
                {"label": "3 years", "value": "3y"},
                {"label": "5 years (recommended)", "value": "5y"},
            ], value="5y", clearable=False, className="period-dropdown"),
            html.Button("Analyze all 50 NSE stocks", id="run-screen", n_clicks=0,
                        className="primary-button"),
        ], className="toolbar"),
        html.Div(UNIVERSE_WARNING, className="notice") if UNIVERSE_WARNING else html.Div(),
        html.Div([
            html.Strong("Forecast limits"),
            html.P("Estimates are not guarantees. The point estimate shrinks historical log-return "
                   "drift toward the risk-free rate. The 90% price range uses the stock's own "
                   "historical 21-session returns. Error charts use monthly holdouts after a "
                   "252-session training period."),
        ], className="notice"),
        html.Div(id="screen-summary"),
        html.Div(id="screen-table", className="table-wrap"),
        html.H2("Per-stock forecast and error charts"),
        html.P("Open any stock row to see its adjusted-price history, forecast, range, "
               "predicted-versus-actual returns and actual-minus-predicted deviation.",
               className="muted"),
        dcc.Loading(html.Div(id="stock-details", className="stock-list"),
                    type="circle", color="#ffca28"),
    ], className="main-content"),
    html.Footer("For research and education. This application cannot eliminate market losses.",
                className="footer"),
], className="page-shell")


@dash_app.callback(
    Output("screen-table", "children"), Output("stock-details", "children"),
    Output("screen-summary", "children"), Input("run-screen", "n_clicks"),
    Input("lookback-period", "value"), prevent_initial_call=True,
)
def update_screen(n_clicks: int, period: str) -> tuple[Any, Any, Any]:
    if not n_clicks:
        return html.Div(), html.Div(), html.Div()
    try:
        return _screen(period)
    except Exception as exc:
        error = html.Div([
            html.Strong("Unable to build the 50-stock screen. "),
            html.Span(f"{type(exc).__name__}: {exc}"),
        ], className="notice notice-error")
        return error, html.Div(), html.Div()


# Dash routes and REST endpoints are mounted on the same ASGI server. Hosts
# that inspect app.py for conventional application variables find these aliases.
app = server
application = server
handler = server

