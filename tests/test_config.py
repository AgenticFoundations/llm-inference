import json
import stat

import pytest

from llm_inference.config import AppError, Config
from llm_inference.runtime import Compose


@pytest.mark.parametrize(
    "key,value",
    [
        ("OLLAMA_KEEP_ALIVE", "-1"),
        ("OLLAMA_PORT", "0"),
        ("EDGE_PORT", "4000"),
        ("MODEL_ALIAS", "../bad"),
        ("OLLAMA_MODEL", "model\nother"),
        ("LITELLM_MASTER_KEY", "short"),
        ("REQUEST_TIMEOUT", "901"),
        ("LITELLM_IMAGE", "ghcr.io/berriai/litellm:main-stable"),
    ],
)
def test_invalid_config_fails_without_starting_anything(config, key, value):
    config.values[key] = value
    with pytest.raises(AppError):
        config.validate()


def test_env_override_and_literal_dollar(config, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "another:tag")
    monkeypatch.setenv("POSTGRES_PASSWORD", "password-with-${NOT_EXPANDED}-and-@/:")
    loaded = Config.load(config.root)
    assert loaded["OLLAMA_MODEL"] == "another:tag"
    assert "${NOT_EXPANDED}" in loaded["POSTGRES_PASSWORD"]
    assert "%24%7BNOT_EXPANDED%7D" in loaded.compose_env()["DATABASE_URL"]


def test_empty_export_is_invalid_and_unsetting_restores_file_value(config, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "")
    with pytest.raises(AppError, match="OLLAMA_MODEL"):
        Config.load(config.root)
    monkeypatch.delenv("OLLAMA_MODEL")
    assert Config.load(config.root)["OLLAMA_MODEL"] == config["OLLAMA_MODEL"]


def test_render_routes_alias_to_selected_native_model(config):
    config.values["OLLAMA_MODEL"] = "different:8b"
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
