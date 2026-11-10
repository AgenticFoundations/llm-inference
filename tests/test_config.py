import json
import stat
from dataclasses import replace

import pytest

from llm_inference.config import Config
from llm_inference.errors import AppError
from llm_inference.models import ModelCatalog
from llm_inference.runtime import Compose


@pytest.mark.parametrize(
    "key,value",
    [
        ("OLLAMA_PORT", "0"),
        ("EDGE_PORT", "4000"),
        ("LITELLM_MASTER_KEY", "short"),
        ("STARTUP_TIMEOUT", "0"),
        ("LITELLM_IMAGE", "ghcr.io/berriai/litellm:main-stable"),
    ],
)
def test_invalid_config_fails_without_starting_anything(config, key, value):
    config.values[key] = value
    with pytest.raises(AppError):
        config.validate()


def test_env_override_and_literal_dollar(config, monkeypatch):
    monkeypatch.setenv("OLLAMA_PORT", "12434")
    monkeypatch.setenv("POSTGRES_PASSWORD", "password-with-${NOT_EXPANDED}-and-@/:")
    loaded = Config.load(config.root)
    assert loaded["OLLAMA_PORT"] == "12434"
    assert "${NOT_EXPANDED}" in loaded["POSTGRES_PASSWORD"]
    assert "%24%7BNOT_EXPANDED%7D" in loaded.compose_env()["DATABASE_URL"]


def test_empty_export_is_invalid_and_unsetting_restores_file_value(config, monkeypatch):
    monkeypatch.setenv("MODELS_CONFIG", "")
    with pytest.raises(AppError, match="MODELS_CONFIG"):
        Config.load(config.root)
    monkeypatch.delenv("MODELS_CONFIG")
    assert Config.load(config.root).catalog == config.catalog


def test_render_routes_alias_to_selected_native_model(config):
    config.catalog = ModelCatalog((replace(config.models[0], model="different:8b"),))
    config.render()
    path = config.root / ".runtime/litellm.yaml"
    rendered = json.loads(path.read_text())
    route = rendered["model_list"][0]
    assert route["model_name"] == "local-coder"
    assert route["litellm_params"]["model"] == "ollama_chat/different:8b"
    assert route["litellm_params"]["api_base"] == "http://host.docker.internal:11434"
    assert route["litellm_params"]["num_ctx"] == 16384
    assert config["LITELLM_MASTER_KEY"] not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_render_uses_each_models_own_context_and_generation_settings(config, second_model):
    config.catalog = ModelCatalog((*config.models, second_model))
    config.validate()
    config.render()
    rendered = json.loads((config.root / ".runtime/litellm.yaml").read_text())
    models = {route["model_name"]: route["litellm_params"] for route in rendered["model_list"]}
    assert set(models) == {"local-coder", "local-coder-bf16"}
    assert models["local-coder"]["model"] == "ollama_chat/" + config.models[0].model
    assert models["local-coder-bf16"]["model"] == "ollama_chat/qwen3.8:27b-mlx-bf16"
    for alias, context, keep_alive, timeout, output in [
        ("local-coder", 16384, "5m", 900, 2048),
        ("local-coder-bf16", 8192, "2m", 120, 1024),
    ]:
        params = models[alias]
        assert params["num_ctx"] == context
        assert params["keep_alive"] == keep_alive
        assert params["timeout"] == timeout
        assert params["max_tokens"] == output
        assert params["api_base"] == "http://host.docker.internal:11434"


def test_old_session_can_be_restored_without_reading_current_files(config):
    legacy = {
        **config.values,
        "MODEL_ALIAS": "old-alias",
        "OLLAMA_MODEL": "old:8b",
        "OLLAMA_ADDITIONAL_MODELS": '{"other-alias":"other:8b"}',
        "OLLAMA_CONTEXT_LENGTH": "16384",
        "OLLAMA_KEEP_ALIVE": "5m",
        "REQUEST_TIMEOUT": "900",
    }
    (config.root / "config/models.json").unlink()
    restored = Config.from_snapshot(config.root, legacy)
    assert restored.model_routes == {"old-alias": "old:8b", "other-alias": "other:8b"}


def test_snapshot_preserves_catalog_when_both_files_are_removed(config):
    snapshot = json.loads(json.dumps(config.snapshot()))
    (config.root / "config/models.json").unlink()
    (config.root / ".env").unlink()
    restored = Config.from_snapshot(config.root, snapshot)
    assert restored.snapshot() == config.snapshot()


def test_catalog_can_be_selected_through_environment(config, monkeypatch):
    custom = config.root / "custom-models.json"
    custom.write_text(json.dumps(config.catalog.to_dict()))
    monkeypatch.setenv("MODELS_CONFIG", str(custom))
    assert Config.load(config.root).catalog == config.catalog


def test_routes_do_not_reparse_json_after_config_load(config, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("The catalog should only be parsed when loading configuration")

    monkeypatch.setattr(json, "loads", unexpected)
    assert config.model_routes == config.model_routes


def test_compose_uses_orbstack_without_inheriting_other_daemon(config, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "production")
    monkeypatch.setenv("COMPOSE_PROFILES", "anything")
    compose = Compose(config)
    assert compose.env["DOCKER_CONTEXT"] == "orbstack"
    assert "DOCKER_HOST" not in compose.env
    assert "COMPOSE_PROFILES" not in compose.env
    assert "--env-file" in compose.command
    assert "/dev/null" in compose.command


def test_no_secrets_in_config_repr(config):
    assert config["LITELLM_MASTER_KEY"] not in repr(config)


def test_retired_snapshot_settings_are_not_forwarded_to_compose(config):
    config.values["RETIRED_SETTING"] = "unused"
    env = config.compose_env()
    assert "RETIRED_SETTING" not in env
