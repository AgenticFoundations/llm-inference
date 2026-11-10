import json
import stat
from contextlib import contextmanager

import httpx
import pytest

from llm_inference import gateway, transport
from llm_inference.errors import AppError
from llm_inference.models import ModelCatalog
from llm_inference.storage import private_json


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


@pytest.mark.parametrize("previous", [None, ["local-coder", "retired-alias"]])
def test_existing_key_gains_model_access_without_rotation(
    config, second_model, monkeypatch, previous
):
    config.catalog = ModelCatalog((*config.models, second_model))
    stored = {"key": "sk-existing", "model": "local-coder"}
    if previous is not None:
        stored["models"] = previous
    path = gateway.key_path(config, "copilot")
    private_json(path, stored)
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json={"data": [{"id": "local-coder"}]})

    monkeypatch.setattr(transport, "request", request)
    assert gateway.client_key(config) == "sk-existing"
    assert [(method, url) for method, url, _ in calls] == [
        ("GET", config.gateway_url + "/v1/models"),
        ("POST", config.gateway_url + "/key/update"),
    ]
    assert calls[1][2]["key"] == config["LITELLM_MASTER_KEY"]
    assert calls[1][2]["payload"] == {
        "key": "sk-existing",
        "models": ["local-coder", "local-coder-bf16"],
    }
    assert json.loads(path.read_text())["models"] == ["local-coder", "local-coder-bf16"]
    assert gateway.client_key(config) == "sk-existing"
    assert len(calls) == 3  # Subsequent calls validate the key without updating it again.


def test_failed_key_scope_update_keeps_existing_key_and_local_metadata(
    config, second_model, monkeypatch
):
    path = gateway.key_path(config, "copilot")
    private_json(path, {"key": "sk-existing", "model": "local-coder"})
    before = path.read_bytes()
    config.catalog = ModelCatalog((*config.models, second_model))

    def request(method, url, **_kwargs):
        if method == "GET":
            return httpx.Response(200)
        assert url.endswith("/key/update")
        return httpx.Response(503, text="private provider details")

    monkeypatch.setattr(transport, "request", request)
    with pytest.raises(AppError, match="Client-key model update failed"):
        gateway.client_key(config)
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["../outside", "", "a/b", "a" * 65])
def test_key_name_cannot_escape_private_directory(config, name):
    with pytest.raises(AppError):
        gateway.key_path(config, name)


def test_auth_check_rejects_an_open_gateway(config, monkeypatch):
    monkeypatch.setattr(transport, "request", lambda *_args, **_kwargs: httpx.Response(200))
    with pytest.raises(AppError, match="absent/invalid"):
        gateway.check_routes(config, "sk-client")


@pytest.mark.parametrize("base_url", [None, "http://localhost:9080"])
@pytest.mark.parametrize("extra", [{}, {"local-coder-bf16": "qwen3.8:27b-mlx-bf16"}])
def test_route_check_accepts_only_authenticated_client_routes(
    config, second_model, monkeypatch, base_url, extra
):
    if extra:
        config.catalog = ModelCatalog((*config.models, second_model))
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
            return httpx.Response(
                200, json={"data": [{"id": alias} for alias in config.model_routes]}
            )
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
        gateway.check_inference(config, "sk-client", config.models[0])
    else:
        with pytest.raises(AppError) as exc:
            gateway.check_inference(config, "sk-client", config.models[0])
        assert "private details" not in str(exc.value)


def test_inference_uses_the_requested_models_alias_and_timeout(config, second_model, monkeypatch):
    from dataclasses import replace

    model = replace(second_model, max_output_tokens=16)
    requests = []
    streams = []

    def request(method, url, **kwargs):
        requests.append(kwargs)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    @contextmanager
    def stream(_method, _url, **kwargs):
        streams.append(kwargs)
        yield httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"ready"}}]}\n\ndata: [DONE]\n'
        )

    monkeypatch.setattr(transport, "request", request)
    monkeypatch.setattr(gateway.httpx, "stream", stream)
    gateway.check_inference(config, "sk-client", model)
    assert requests[0]["payload"]["model"] == "local-coder-bf16"
    assert requests[0]["payload"]["max_tokens"] == 16
    assert requests[0]["timeout"] == streams[0]["timeout"] == 120
    assert streams[0]["json"]["model"] == "local-coder-bf16"
