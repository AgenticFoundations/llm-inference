import json
import stat
from types import SimpleNamespace

import httpx
import pytest

from llm_inference import controller as module
from llm_inference.controller import Controller
from llm_inference.errors import AppError
from llm_inference.models import ModelCatalog
from llm_inference.runtime import lifecycle_lock
from llm_inference.storage import private_json


@pytest.fixture
def harness(config, monkeypatch):
    events = []

    class FakeCompose:
        fail_up = False
        fail_down = False

        def __init__(self, current):
            self.config = current

        def preflight(self):
            events.append("preflight")

        def call(self, *args, **kwargs):
            events.append(args)
            if args == ("ps", "--status", "running", "--services"):
                return "db\ngateway\nedge\n"
            if args[0] == "up" and self.fail_up:
                raise AppError("simulated partial Compose startup")
            if args[0] == "down" and self.fail_down:
                raise AppError("simulated daemon failure")
            return ""

    monkeypatch.setattr(module, "Compose", FakeCompose)
    monkeypatch.setattr(module, "port_in_use", lambda _: False)
    monkeypatch.setattr(
        module.ollama,
        "unload_models",
        lambda current: events.append(("unload-model", current.models[0].model)),
    )
    monkeypatch.setattr(Controller, "ready", staticmethod(lambda _: True))
    monkeypatch.setattr(module.gateway, "client_key", lambda _: "sk-client")
    monkeypatch.setattr(module.gateway, "check_routes", lambda *_: events.append("auth-checked"))
    monkeypatch.setattr(
        module.transport,
        "request",
        lambda *_: httpx.Response(
            200, json={"models": [{"name": model.model} for model in config.models]}
        ),
    )
    return SimpleNamespace(
        controller=Controller(config.root),
        events=events,
        compose=FakeCompose,
        config=config,
    )


def compose_events(harness, command):
    return [e for e in harness.events if isinstance(e, tuple) and e[0] == command]


def test_startup_order_idempotence_and_shutdown(harness):
    app = harness.controller
    app.up()
    assert app.state()["phase"] == "ready"
    assert stat.S_IMODE(app.state_path.stat().st_mode) == 0o600
    assert "auth-checked" in harness.events
    copies = compose_events(harness, "cp")
    assert [entry[-1] for entry in copies] == [
        "gateway:/app/config.yaml",
        "edge:/etc/nginx/nginx.conf",
    ]
    startup = compose_events(harness, "up")[0]
    assert all(harness.events.index(entry) < harness.events.index(startup) for entry in copies)
    before = len(compose_events(harness, "up"))
    app.up()
    assert len(compose_events(harness, "up")) == before
    app.down()
    assert not app.state_path.exists()
    assert ("unload-model", harness.config.models[0].model) in harness.events
    assert harness.events.index(("down", "--timeout", "10", "--remove-orphans")) < (
        harness.events.index(("unload-model", harness.config.models[0].model))
    )
    assert ("down", "--timeout", "10", "--remove-orphans") in harness.events
    app.down()  # Repeated stop is harmless.


def test_missing_model_leaves_existing_server_and_models_alone(harness, monkeypatch):
    monkeypatch.setattr(
        module.transport, "request", lambda *_: httpx.Response(200, json={"models": []})
    )
    with pytest.raises(AppError, match="not installed"):
        harness.controller.up()
    assert not any(isinstance(e, tuple) and e[0] == "unload-model" for e in harness.events)
    assert not any(isinstance(e, tuple) and e[0] == "up" for e in harness.events)
    assert not harness.controller.state_path.exists()


def test_partial_compose_startup_is_cleaned_up(harness):
    harness.compose.fail_up = True
    with pytest.raises(AppError, match="partial Compose"):
        harness.controller.up()
    assert ("down", "--timeout", "10", "--remove-orphans") in harness.events
    assert not any(isinstance(e, tuple) and e[0] == "unload-model" for e in harness.events)
    assert not harness.controller.state_path.exists()


def test_interrupt_during_startup_rolls_back(harness, monkeypatch):
    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.gateway, "check_routes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        harness.controller.up()
    assert not any(isinstance(e, tuple) and e[0] == "unload-model" for e in harness.events)
    assert not harness.controller.state_path.exists()


def test_stop_survives_env_deletion_and_retains_failed_cleanup(harness):
    harness.controller.up()
    (harness.config.root / ".env").unlink()
    harness.compose.fail_down = True
    with pytest.raises(AppError, match="Session retained"):
        harness.controller.down()
    assert harness.controller.state()["compose_started"] is True
    assert harness.controller.state()["unload_on_stop"] is True
    assert not any(isinstance(e, tuple) and e[0] == "unload-model" for e in harness.events)
    harness.compose.fail_down = False
    harness.controller.down()
    assert not harness.controller.state_path.exists()


def test_existing_ollama_listener_is_used_without_a_second_process(harness, monkeypatch):
    monkeypatch.setattr(module, "port_in_use", lambda port: port == 11434)
    harness.controller.up()
    assert harness.controller.state()["phase"] == "ready"
    assert "ollama" not in harness.controller.state()
    harness.controller.down()


def test_missing_ollama_does_not_launch_or_install_anything(harness, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(module.transport, "request", unavailable)
    with pytest.raises(AppError, match="existing Ollama server is unavailable"):
        harness.controller.up()
    assert not harness.events
    assert not harness.controller.state_path.exists()


def test_lifecycle_uses_only_project_compose_operations(harness):
    harness.controller.up()
    harness.controller.down()
    compose_commands = {
        event[0]
        for event in harness.events
        if isinstance(event, tuple) and event[0] != "unload-model"
    }
    assert compose_commands <= {"ps", "create", "cp", "up", "down", "exec"}


def test_changed_config_requires_restart(harness, monkeypatch):
    harness.controller.up()
    path = harness.config.root / "config/models.json"
    data = json.loads(path.read_text())
    data["models"][0]["model"] = "another:tag"
    path.write_text(json.dumps(data))
    with pytest.raises(AppError, match="prior session"):
        harness.controller.up()
    harness.controller.down()


def test_concurrent_mutations_fail_cleanly(tmp_path):
    with lifecycle_lock(tmp_path), pytest.raises(AppError, match="Another lifecycle"):
        with lifecycle_lock(tmp_path):
            pytest.fail("must not acquire a second lock")


def test_resolve_config_uses_file_before_start_and_snapshot_after_start(harness, monkeypatch):
    app = harness.controller
    assert app.resolve_config().values == harness.config.values
    app.up()
    path = harness.config.root / "config/models.json"
    data = json.loads(path.read_text())
    data["models"][0]["model"] = "another:tag"
    path.write_text(json.dumps(data))
    assert app.resolve_config().models[0].model == harness.config.models[0].model
    assert app.active_config().models[0].model == harness.config.models[0].model
    (harness.config.root / ".env").unlink()
    path.unlink()
    assert app.resolve_config().values == harness.config.values
    assert app.active_config().values == harness.config.values


def test_status_without_session_contacts_no_services(harness, capsys):
    harness.controller.status()
    assert "off (no managed session)" in capsys.readouterr().out
    assert not harness.events


def test_status_reports_saved_routing_and_configuration_drift(harness, monkeypatch, capsys):
    harness.controller.up()
    capsys.readouterr()
    harness.events.clear()
    path = harness.config.root / "config/models.json"
    data = json.loads(path.read_text())
    data["models"][0]["model"] = "another:tag"
    path.write_text(json.dumps(data))
    harness.controller.status()
    output = capsys.readouterr().out
    assert f"local-coder -> {harness.config.models[0].model}" in output
    assert f"Loaded in memory: {harness.config.models[0].model}" in output
    assert "Configuration differs" in output
    assert harness.events == [("ps", "--all")]


def test_status_survives_missing_env_and_unavailable_ollama(harness, monkeypatch, capsys):
    harness.controller.up()
    capsys.readouterr()
    (harness.config.root / ".env").unlink()

    def unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("private connection details")

    monkeypatch.setattr(module.transport, "request", unavailable)
    harness.controller.status()
    output = capsys.readouterr().out
    assert "Loaded in memory: unavailable" in output
    assert "showing the saved session configuration" in output
    assert "private connection details" not in output


def configure_two_models(harness, second_model):
    harness.config.catalog = ModelCatalog((*harness.config.models, second_model))
    path = harness.config.root / "config/models.json"
    path.write_text(json.dumps(harness.config.catalog.to_dict()))


@pytest.mark.parametrize("failed", [None, "local-coder", "local-coder-bf16"])
def test_check_smokes_every_model_and_reports_any_failures(
    harness, second_model, monkeypatch, capsys, failed
):
    configure_two_models(harness, second_model)
    harness.controller.up()
    private_json(module.gateway.key_path(harness.config, "copilot"), {"key": "sk-client"})
    tested = []

    def check(_config, _key, model):
        tested.append(model.alias)
        if model.alias == failed:
            raise AppError("private upstream error")

    monkeypatch.setattr(module.gateway, "check_inference", check)
    if failed:
        with pytest.raises(AppError, match=failed):
            harness.controller.check()
    else:
        harness.controller.check()
    assert tested == ["local-coder", "local-coder-bf16"]
    assert "private upstream error" not in capsys.readouterr().out


def test_check_can_target_one_alias_without_loading_the_other(harness, second_model, monkeypatch):
    configure_two_models(harness, second_model)
    harness.controller.up()
    private_json(module.gateway.key_path(harness.config, "copilot"), {"key": "sk-client"})
    tested = []
    monkeypatch.setattr(
        module.gateway, "check_inference", lambda _config, _key, model: tested.append(model.alias)
    )
    harness.controller.check("local-coder-bf16")
    assert tested == ["local-coder-bf16"]


def test_copilot_export_uses_catalog_budgets_without_contacting_services(harness, second_model):
    configure_two_models(harness, second_model)
    harness.controller.copilot_config()
    path = harness.config.root / ".runtime/copilot-models.json"
    data = json.loads(path.read_text())
    models = {model["id"]: model for model in data["models"]}
    assert models["local-coder"]["contextWindow"] == 16384
    assert models["local-coder-bf16"]["contextWindow"] == 8192
    assert models["local-coder-bf16"]["maxOutputTokens"] == 1024
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "apiKey" not in path.read_text()
    assert not harness.events


def test_model_start_requires_explicit_alias_and_uses_that_models_settings(
    harness, second_model, monkeypatch
):
    configure_two_models(harness, second_model)
    loaded = []
    monkeypatch.setattr(module.ollama, "load_model", lambda _config, model: loaded.append(model))
    harness.controller.model_start("local-coder-bf16")
    assert loaded == [second_model]
