# NSE Stock Risk & Return Calculator

The app is a Dash dashboard mounted on the same FastAPI ASGI server as the REST API. It screens the 50 current NIFTY 50 constituents, shows a one-month prediction column, and gives each ticker a forecast chart plus a walk-forward predicted-versus-actual deviation chart. NASDAQ support has been removed.

## Run locally

Use Python 3.12 or newer from this folder.

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for API documentation. The API liveness endpoint is GET /health.

app.py exports app, application, and handler as the shared FastAPI application, for hosts that search for a conventional top-level app variable. Dash routes are mounted on that server.

## Forecast interpretation

The one-month horizon is 21 trading sessions. Point estimates use a posterior log-return drift shrunk toward the risk-free rate. The 90% price range is based on the historical distribution of overlapping 21-session log returns, shifted to the estimated drift. The error chart is a walk-forward evaluation: the model starts with 252 sessions, forecasts the next 21 sessions, then advances by 21 sessions. Its mean absolute error and mean bias summarize those historical holdouts.

The displayed range and point estimate are model outputs, not calibrated guarantees. Market prices can gap, historical behavior can change, and past accuracy does not ensure future accuracy. NSE constituents are current constituents, so backtests can have survivorship bias. Forecasting does not remove investment risk.

## API

- GET /v1/markets - NSE benchmark, currency and 50-stock universe.
- POST /v1/screen - analyze up to 50 NSE tickers; omitting tickers uses the current NIFTY 50 list.
- POST /v1/metrics - risk, return and one-month forecast metrics for a ticker.
- POST /v1/history - adjusted prices for a ticker and the NIFTY benchmark.
- POST /v1/simulation - lagged-volatility risk-control simulation.

The accepted market is Indian Market (NSE). Tickers must be Yahoo Finance NSE symbols ending in .NS. Periods are 1y, 3y, or 5y. Set STOCK_API_KEY to require X-API-Key on /v1/* endpoints.

Example request body:
    {"market": "Indian Market (NSE)", "tickers": ["RELIANCE.NS", "INFY.NS"], "period": "5y"}

## Docker and deployment

Build and run from this folder.

    docker build -t nse-stock-risk .
    docker run --rm -p 8000:8000 nse-stock-risk

The included Dockerfile starts uvicorn app:app on 0.0.0.0 and uses the platform-provided PORT environment variable. For a Git-based web service, configure the start command as uvicorn app:app --host 0.0.0.0 --port $PORT (or deploy the included Dockerfile). Keep the REST API key in the host's secret environment settings.

Market data is downloaded from Yahoo Finance through yfinance, and the constituent list comes from NSE Indices. Provider availability, ticker coverage and request limits can affect which of the 50 rows have usable prices on a given run.
