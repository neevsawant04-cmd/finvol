"""Small standard-library HTTP client used by Streamlit when a backend is set."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BackendRequestError(RuntimeError):
    """Readable error raised when the configured backend cannot serve a request."""


def post_json(
    base_url: str,
    path: str,
    payload: dict,
    api_key: str = "",
    timeout: int = 240,
) -> dict:
    """POST JSON to the configured API and return its decoded object response."""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(
        url,
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            detail = error_body.get("detail", error_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = f"HTTP {exc.code}"
        raise BackendRequestError(f"Backend returned HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise BackendRequestError(f"Could not reach the stock backend at {base_url}: {reason}") from exc

    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendRequestError("The stock backend returned an invalid JSON response.") from exc
    if not isinstance(decoded, dict):
        raise BackendRequestError("The stock backend response must be a JSON object.")
    return decoded

