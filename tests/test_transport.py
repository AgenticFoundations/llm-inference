import httpx
import pytest

from llm_inference import transport
from llm_inference.config import AppError


@pytest.mark.parametrize("status", [401, 503])
def test_http_failures_report_status_without_leaking_response_body(status):
    response = httpx.Response(status, text="private provider response containing a secret")
    with pytest.raises(AppError) as exc:
        transport.ok(response, "Model loading")
    assert str(exc.value) == f"Model loading failed (HTTP {status})."
