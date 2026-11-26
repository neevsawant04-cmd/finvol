# Stock Risk & Return API

The backend is a separate FastAPI service. It shares the same market-data loader and risk calculations as the Streamlit dashboard. The dashboard uses it when `STOCK_API_URL` is configured; without that setting, it continues to run calculations in the Streamlit process.

## Run locally

From this folder, install the API dependencies and start the service:

```powershell
py -3.12 -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
$env:STOCK_API_KEY = "choose-a-long-random-secret"
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. `GET /health` is an unauthenticated liveness check. All `/v1/*` routes require `X-API-Key` when the server has `STOCK_API_KEY` set.

## Endpoints

- `GET /v1/markets` — market names, benchmark symbols, currencies, and built-in ticker counts.
- `POST /v1/screen` — analyze a ticker list (up to 550 symbols) for a market and lookback.
- `POST /v1/metrics` — metrics for one ticker.
- `POST /v1/history` — adjusted price history for one ticker and its benchmark.
- `POST /v1/simulation` — walk-forward volatility-target simulation with configurable target volatility and transaction costs.

Example screen request:

```json
{
  "market": "NASDAQ",
  "tickers": ["AAPL", "MSFT"],
  "period": "3y",
  "annual_risk_free_rate": 0.0485
}
```

The market must be `NASDAQ` or `Indian Market (NSE)`. The period must be `1y`, `3y`, or `5y`. If no ticker list is provided, the backend uses the built-in universe (the current NIFTY 500 list for NSE, with a fallback if that list cannot be fetched).

## Connect Streamlit to a deployed backend

Deploy this folder as a separate Docker web service using the included `Dockerfile`. The service listens on the platform-provided `PORT` (default 8000). Set `STOCK_API_KEY` as a secret on the API service. Then add these secrets to the Streamlit app:

```toml
STOCK_API_URL = "https://your-api-host.example"
STOCK_API_KEY = "the-same-long-random-secret"
```

The browser never receives the API key; Streamlit sends it in a server-to-server request header. Streamlit Community Cloud runs the Streamlit process, so the API needs its own web-service deployment. Keep `/docs` private or disable it at the hosting layer if you need to hide the interactive schema page.

Market history is cached in process memory for one hour and the NIFTY constituent list for one day. Cache entries are lost when the service restarts. The service uses Yahoo Finance through `yfinance`; data availability and request limits depend on that provider. This backend does not store user accounts, portfolios, or trade records.

