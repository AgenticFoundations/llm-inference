"""Gateway lifecycle and model residency using the user's existing Ollama server."""

import json
from pathlib import Path
from typing import Literal, TypedDict, cast

import httpx

from . import gateway, ollama, transport
from .config import AppError, Config
from .runtime import (
    Compose,
    port_in_use,
)
from .storage import private_json


class SessionState(TypedDict):
    config: dict[str, str]
    compose_started: bool
    unload_on_stop: bool
    phase: Literal["starting", "ready", "stopping"]


class Controller:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / ".runtime/session.json"

    def state(self) -> SessionState | None:
        if not self.state_path.exists():
            return None
        try:
            return cast(SessionState, json.loads(self.state_path.read_text()))
        except (ValueError, OSError) as exc:
            raise AppError("Cannot read .runtime/session.json; preserve it for recovery.") from exc

    def save(self, state: SessionState) -> None:
        private_json(self.state_path, state)

    def active_config(self) -> Config:
        state = self.state()
        if not state or state["phase"] != "ready":
            raise AppError("Application is not ready. Run make up; use make status to inspect it.")
        return Config(self.root, state["config"])

    def doctor(self) -> None:
        config = Config.load(self.root)
        Compose(config).preflight()
        ollama.require_model(config)
        print("Configuration, existing Ollama server, selected model, and OrbStack are available.")

    def up(self) -> None:
        config = Config.load(self.root)
        previous = self.state()
        if previous:
            if (
                previous["phase"] == "ready"
                and previous["config"] == config.values
                and self.ready(config.ollama_url + "/api/tags")
                and self.ready(config.edge_url + "/healthz")
                and self.ready(config.gateway_url + "/health/readiness")
                and self.services_running(config)
            ):
                print("Application is already running. Use make restart to restart it.")
                return
            raise AppError("A prior session exists. Run make down or make restart to reconcile it.")
        for key in ("GATEWAY_PORT", "EDGE_PORT"):
            if port_in_use(config.number(key)):
                raise AppError(
                    f"{key}={config[key]} is already in use. Quit the independent service "
                    "or change the port in .env. Existing processes will not be stopped."
                )
        ollama.require_model(config)
        compose = Compose(config)
        compose.preflight()
        if compose.call("ps", "--all", "--quiet").strip():
            raise AppError("Compose project llm-inference already has unmanaged containers.")
        config.render()
        state: SessionState = {
            "config": config.values,
            "compose_started": False,
            "unload_on_stop": False,
            "phase": "starting",
        }
        self.save(state)
        try:
            print(f"Using your existing Ollama server at {config.ollama_url}.", flush=True)
            print("Starting gateway, database, and ingress on OrbStack...", flush=True)
            # Save before Compose: a failed command may leave partially created containers.
            state["compose_started"] = True
            self.save(state)
            # Upload configs through the Docker API rather than bind-mounting Documents.
            compose.call("create", "db", "gateway", "edge", timeout=120)
            compose.call("cp", str(self.root / ".runtime/litellm.yaml"), "gateway:/app/config.yaml")
            compose.call("cp", str(self.root / "config/nginx.conf"), "edge:/etc/nginx/nginx.conf")
            compose.call(
                "up",
                "-d",
                "--wait",
                "--wait-timeout",
                config["STARTUP_TIMEOUT"],
                "db",
                "gateway",
                "edge",
                timeout=config.number("STARTUP_TIMEOUT") + 600,
                capture=False,
            )
            self.check_bridge(config, compose)
            key = gateway.client_key(config)
            gateway.check_routes(config, key)
            state["phase"] = "ready"
            state["unload_on_stop"] = True
            self.save(state)
        except BaseException:
            print("Startup did not finish; stopping application resources...", flush=True)
            try:
                self.down()
            except AppError as cleanup_error:
                print(f"Cleanup needs attention: {cleanup_error}")
            raise
        print(f"Ready: {config.edge_url}/v1 — model {config['MODEL_ALIAS']}")
        print("Client key: make key NAME=copilot. Stop gateway and unload model: make down.")

    @staticmethod
    def services_running(config: Config) -> bool:
        running = set(Compose(config).call("ps", "--status", "running", "--services").split())
        return {"db", "gateway", "edge"} <= running

    @staticmethod
    def ready(url: str) -> bool:
        try:
            return transport.request("GET", url, timeout=2).is_success
        except httpx.RequestError:
            return False

    @staticmethod
    def check_bridge(config: Config, compose: Compose) -> None:
        url = f"http://host.docker.internal:{config['OLLAMA_PORT']}/api/tags"
        code = (
            "import json, urllib.request; "
            f"data=json.load(urllib.request.urlopen({url!r},timeout=5)); "
            f"assert {config['OLLAMA_MODEL']!r} in [m['name'] for m in data['models']]"
        )
        try:
            compose.call("exec", "-T", "gateway", "python", "-c", code)
        except AppError as exc:
            raise AppError(
                "The gateway cannot reach native Ollama through host.docker.internal. "
                "Check OrbStack host networking. Ollama remains bound to loopback."
            ) from exc

    def down(self) -> None:
        state = self.state()
        if not state:
            print("No application session to stop.")
            return
        config = Config(self.root, state["config"])
        errors = []
        model_unloaded = False
        state["phase"] = "stopping"
        self.save(state)
        if state["compose_started"]:
            compose = Compose(config)
            try:
                compose.call("down", "--timeout", "10", "--remove-orphans", timeout=90)
                state["compose_started"] = False
            except AppError as exc:
                errors.append(str(exc))
        # Do not unload while a failed gateway shutdown could still accept new requests.
        if not state["compose_started"] and state.get("unload_on_stop", False):
            try:
                ollama.unload_model(config)
                state["unload_on_stop"] = False
                model_unloaded = True
            except (httpx.RequestError, AppError) as exc:
                detail = str(exc) if isinstance(exc, AppError) else "Ollama is unavailable."
                errors.append(f"Could not unload the selected model: {detail}")
        if errors:
            self.save(state)
            raise AppError(" ".join(errors) + " Session retained; retry make down.")
        self.state_path.unlink()
        print("Gateway stopped. Your existing Ollama server stays running.")
        if model_unloaded:
            print(f"Selected model unloaded: {config['OLLAMA_MODEL']}.")
        print("Downloaded models, client keys, and database volume are preserved.")

    def status(self) -> None:
        state = self.state()
        if not state:
            print("Application: off (no managed session).")
            return
        config = Config(self.root, state["config"])
        print(f"Application: {state['phase']}")
        print("Ollama: existing external server (never started or stopped by this project)")
        print(f"Routing: {config['MODEL_ALIAS']} -> {config['OLLAMA_MODEL']}")
        try:
            response = transport.ok(
                transport.request("GET", config.ollama_url + "/api/ps"), "Loaded models"
            )
            names = [model["name"] for model in response.json()["models"]]
            print("Loaded in memory: " + (", ".join(names) or "none"))
        except (httpx.RequestError, AppError):
            print("Loaded in memory: unavailable")
        if state["compose_started"]:
            Compose(config).call("ps", "--all", capture=False)
        try:
            if Config.load(self.root).values != config.values:
                print(".env differs from this running session. Apply it with make restart.")
        except AppError:
            print(
                "Current .env is unavailable or invalid; showing the saved session configuration."
            )

    def logs(self, service: str | None) -> None:
        config = self.active_config()
        args = ["logs", "--follow", "--tail", "100"]
        if service:
            args.append(service)
        Compose(config).call(*args, timeout=None, capture=False)

    def resolve_config(self) -> Config:
        state = self.state()
        return Config(self.root, state["config"]) if state else Config.load(self.root)

    def model_start(self) -> None:
        config = self.resolve_config()
        ollama.load_model(config)
        print(f"Loaded {config['OLLAMA_MODEL']} using your existing Ollama server.")

    def model_stop(self) -> None:
        config = self.resolve_config()
        ollama.unload_model(config)
        print(f"Unloaded {config['OLLAMA_MODEL']}. Your Ollama server stays running.")

    def check(self) -> None:
        config = self.active_config()
        path = gateway.key_path(config, "copilot")
        if not path.exists():
            raise AppError("Create the check client first: make key NAME=copilot.")
        key = json.loads(path.read_text())["key"]
        gateway.check_routes(config, key)
        print("Authentication, model discovery, and client route restrictions passed.", flush=True)
        self.check_bridge(config, Compose(config))
        print("Testing inference and streaming (this loads the model)...", flush=True)
        gateway.check_inference(config, key)
        print("Inference and streaming passed.")
