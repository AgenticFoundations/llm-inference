import httpx
import pytest

from llm_inference import ollama
from llm_inference.errors import AppError
from llm_inference.models import ModelCatalog


def test_unload_targets_only_configured_model_and_keeps_other_models(config, monkeypatch):
    loaded = [config.models[0].model, "another:8b"]
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return httpx.Response(200, json={"models": [{"name": name} for name in loaded]})
        assert url.endswith("/api/generate")
        assert kwargs["payload"] == {
            "model": config.models[0].model,
            "stream": False,
            "keep_alive": 0,
        }
        loaded.remove(config.models[0].model)
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.unload_model(config, config.models[0].model)
    assert loaded == ["another:8b"]
    assert sum(method == "POST" for method, _, _ in calls) == 1


def test_unloaded_model_is_not_accidentally_loaded_by_shutdown(config, monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append(method)
        return httpx.Response(200, json={"models": []})

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.unload_model(config, config.models[0].model)
    assert calls == ["GET"]


def test_explicit_preload_uses_existing_api_without_generating_a_prompt(config, monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json={"models": [{"name": config.models[0].model}]})

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.load_model(config, config.models[0])
    body = calls[-1][2]["payload"]
    assert calls[-1][1] == config.ollama_url + "/api/generate"
    assert body["model"] == config.models[0].model
    assert body["options"]["num_ctx"] == 16384
    assert body["keep_alive"] == "5m"
    assert "prompt" not in body


def test_startup_requires_all_configured_models_without_loading_them(
    config, second_model, monkeypatch
):
    config.catalog = ModelCatalog((*config.models, second_model))
    calls = []

    def request(method, url, **_kwargs):
        calls.append((method, url))
        return httpx.Response(200, json={"models": [{"name": config.models[0].model}]})

    monkeypatch.setattr(ollama.transport, "request", request)
    with pytest.raises(AppError, match="qwen3.8:27b-mlx-bf16.*not installed"):
        ollama.require_model(config)
    assert calls == [("GET", config.ollama_url + "/api/tags")]


def test_unload_all_configured_models_preserves_unrelated_models(config, second_model, monkeypatch):
    second = "qwen3.8:27b-mlx-bf16"
    config.catalog = ModelCatalog((*config.models, second_model))
    loaded = [config.models[0].model, second, "unrelated:8b"]
    unloaded = []

    def request(method, url, **kwargs):
        if method == "GET":
            return httpx.Response(200, json={"models": [{"name": name} for name in loaded]})
        payload = kwargs["payload"]
        assert payload["keep_alive"] == 0
        unloaded.append(payload["model"])
        loaded.remove(payload["model"])
        return httpx.Response(200)

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.unload_models(config)
    assert unloaded == [config.models[0].model, second]
    assert loaded == ["unrelated:8b"]


def test_bulk_unload_retries_remaining_models_and_reports_failed_tags(
    config, second_model, monkeypatch
):
    second = "qwen3.8:27b-mlx-bf16"
    config.catalog = ModelCatalog((*config.models, second_model))
    attempts = []

    def unload(_config, model):
        attempts.append(model)
        if model == config.models[0].model:
            raise httpx.ConnectError("private details")

    monkeypatch.setattr(ollama, "unload_model", unload)
    with pytest.raises(AppError) as exc:
        ollama.unload_models(config)
    assert attempts == [config.models[0].model, second]
    assert config.models[0].model in str(exc.value)
    assert "private details" not in str(exc.value)


def test_explicit_unload_rejects_unconfigured_model_before_any_http(config, monkeypatch):
    def request(*_args, **_kwargs):
        raise AssertionError("Unconfigured model must not reach Ollama")

    monkeypatch.setattr(ollama.transport, "request", request)
    with pytest.raises(AppError, match="outside this project's configuration"):
        ollama.unload_model(config, "unrelated:8b")


@pytest.mark.parametrize(
    "data", [None, {}, {"models": {}}, {"models": [{}]}, {"models": [{"name": 42}]}]
)
def test_loaded_models_validates_response_shape(config, monkeypatch, data):
    monkeypatch.setattr(
        ollama.transport, "request", lambda *_args, **_kwargs: httpx.Response(200, json=data)
    )
    with pytest.raises(AppError, match="model|JSON"):
        ollama.loaded_models(config)


def test_malformed_model_response_does_not_skip_later_unload_attempts(
    config, second_model, monkeypatch
):
    config.catalog = ModelCatalog((*config.models, second_model))
    responses = iter(
        [
            httpx.Response(200, json={"models": [{}]}),
            httpx.Response(200, json={"models": [{"name": second_model.model}]}),
            httpx.Response(200, json={"done": True}),
            httpx.Response(200, json={"models": []}),
        ]
    )
    unloaded = []

    def request(method, _url, **kwargs):
        if method == "POST":
            unloaded.append(kwargs["payload"]["model"])
        return next(responses)

    monkeypatch.setattr(ollama.transport, "request", request)
    with pytest.raises(AppError, match=config.models[0].model):
        ollama.unload_models(config)
    assert unloaded == [second_model.model]
