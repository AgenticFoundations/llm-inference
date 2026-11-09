import httpx

from llm_inference import ollama


def test_unload_targets_only_configured_model_and_keeps_other_models(config, monkeypatch):
    loaded = [config["OLLAMA_MODEL"], "another:8b"]
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return httpx.Response(200, json={"models": [{"name": name} for name in loaded]})
        assert url.endswith("/api/generate")
        assert kwargs["payload"] == {
            "model": config["OLLAMA_MODEL"],
            "stream": False,
            "keep_alive": 0,
        }
        loaded.remove(config["OLLAMA_MODEL"])
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.unload_model(config)
    assert loaded == ["another:8b"]
    assert sum(method == "POST" for method, _, _ in calls) == 1


def test_unloaded_model_is_not_accidentally_loaded_by_shutdown(config, monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append(method)
        return httpx.Response(200, json={"models": []})

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.unload_model(config)
    assert calls == ["GET"]


def test_explicit_preload_uses_existing_api_without_generating_a_prompt(config, monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json={"models": [{"name": config["OLLAMA_MODEL"]}]})

    monkeypatch.setattr(ollama.transport, "request", request)
    ollama.load_model(config)
    body = calls[-1][2]["payload"]
    assert calls[-1][1] == config.ollama_url + "/api/generate"
    assert body["model"] == config["OLLAMA_MODEL"]
    assert body["options"]["num_ctx"] == 16384
    assert body["keep_alive"] == "5m"
    assert "prompt" not in body
