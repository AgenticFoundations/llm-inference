import json
import stat
from contextlib import contextmanager

import httpx
import pytest

from llm_inference import gateway, transport
from llm_inference.config import AppError


def test_key_is_scoped_reused_and_revoked(config, monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json={"key": "sk-client"})

    monkeypatch.setattr(transport, "request", request)
    assert gateway.client_key(config) == "sk-client"
    assert calls[0][2]["payload"]["models"] == ["local-coder"]
    path = gateway.key_path(config, "copilot")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert gateway.client_key(config) == "sk-client"
    assert len(calls) == 2
    gateway.revoke_key(config, "copilot")
    assert calls[-1][2]["payload"] == {"keys": ["sk-client"]}
    assert not path.exists()


@pytest.mark.parametrize("name", ["../outside", "", "a/b", "a" * 65])
def test_key_name_cannot_escape_private_directory(config, name):
    with pytest.raises(AppError):
        gateway.key_path(config, name)


def test_auth_check_rejects_an_open_gateway(config, monkeypatch):
    monkeypatch.setattr(transport, "request", lambda *_args, **_kwargs: httpx.Response(200))
    with pytest.raises(AppError, match="absent/invalid"):
        gateway.check_routes(config, "sk-client")


@pytest.mark.parametrize("base_url", [None, "http://localhost:9080"])
def test_route_check_accepts_only_authenticated_client_routes(config, monkeypatch, base_url):
    base = base_url or config.edge_url
    blocked = {"/ui", "/key/generate", "/config/yaml", "/model/new", "/api/tags"}
    calls = []

    def request(method, url, *, key=None, **_kwargs):
        path = url.removeprefix(base)
        assert url.startswith(base + "/")
        calls.append((method, path, key))
        if method == "GET" and path == "/v1/models":
            if key != "sk-client":
                return httpx.Response(401)
            return httpx.Response(200, json={"data": [{"id": config["MODEL_ALIAS"]}]})
        if method == "POST" and path in blocked:
            return httpx.Response(404)
        if (method, path) in (("POST", "/v1/models"), ("GET", "/v1/chat/completions")):
            return httpx.Response(405)
        raise AssertionError(f"Unexpected route check: {method} {path}")

    monkeypatch.setattr(transport, "request", request)
    gateway.check_routes(config, "sk-client", base_url)
    assert calls[:3] == [
        ("GET", "/v1/models", None),
        ("GET", "/v1/models", "sk-deliberately-invalid"),
        ("GET", "/v1/models", "sk-client"),
    ]
    assert {path for method, path, _ in calls if method == "POST" and path in blocked} == blocked
    assert calls[-2:] == [
        ("POST", "/v1/models", "sk-client"),
        ("GET", "/v1/chat/completions", "sk-client"),
    ]


@pytest.mark.parametrize(
    "lines,passes",
    [
        (
            ["data: " + json.dumps({"choices": [{"delta": {"content": "ready"}}]}), "data: [DONE]"],
            True,
        ),
        (["data: " + json.dumps({"error": {"message": "private details"}})], False),
        (["data: {invalid JSON: private details"], False),
        (["data: null"], False),
        (["data: 42"], False),
        (["data: " + json.dumps("private details")], False),
        (["data: " + json.dumps(["private details"])], False),
        (["data: " + json.dumps({"choices": [{"delta": {}}]})], False),
        (["data: [DONE]"], False),
    ],
)
def test_stream_requires_choices_and_done(config, monkeypatch, lines, passes):
    monkeypatch.setattr(
        transport,
        "request",
        lambda *_args, **_kwargs: httpx.Response(
            200, json={"choices": [{"message": {"content": "ready"}}]}
        ),
    )

    @contextmanager
    def stream(*_args, **_kwargs):
        yield httpx.Response(200, text="\n".join(lines))

    monkeypatch.setattr(gateway.httpx, "stream", stream)
    if passes:
        gateway.check_inference(config, "sk-client")
    else:
        with pytest.raises(AppError) as exc:
            gateway.check_inference(config, "sk-client")
        assert "private details" not in str(exc.value)
