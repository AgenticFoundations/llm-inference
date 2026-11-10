"""Shared HTTP requests and status checks; response bodies stay out of errors."""

import httpx

from .errors import AppError


def request(
    method: str,
    url: str,
    *,
    key: str | None = None,
    timeout: float = 5,
    payload: object = None,
) -> httpx.Response:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.request(
        method, url, headers=headers, timeout=timeout, trust_env=False, json=payload
    )


def ok(response: httpx.Response, operation: str) -> httpx.Response:
    if not response.is_success:
        raise AppError(f"{operation} failed (HTTP {response.status_code}).")
    return response
