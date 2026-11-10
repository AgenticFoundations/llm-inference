"""Control model residency through the user's existing Ollama server."""

import httpx

from . import transport
from .config import Config
from .errors import AppError
from .models import ModelSpec
from .runtime import wait_for

UNLOAD_TIMEOUT_SECONDS = 30


def model_names(response: httpx.Response) -> set[str]:
    try:
        data = response.json()
    except ValueError:
        raise AppError("Ollama returned invalid JSON for its model list.") from None
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise AppError("Ollama returned a malformed model list.")
    names = set()
    for item in data["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise AppError("Ollama returned a malformed model entry.")
        names.add(item["name"])
    return names


def require_model(config: Config) -> None:
    try:
        response = transport.ok(
            transport.request("GET", config.ollama_url + "/api/tags"), "Ollama model list"
        )
    except httpx.RequestError as exc:
        raise AppError(
            f"Your existing Ollama server is unavailable at {config.ollama_url}. "
            "Open your installed Ollama app yourself; this project never launches a server."
        ) from exc
    names = model_names(response)
    missing = set(config.model_routes.values()) - names
    if missing:
        available = ", ".join(sorted(names)) or "none"
        raise AppError(
            f"Model {', '.join(sorted(missing))} is not installed. Available: {available}. "
            "Configure exact installed model tags. No automatic downloads."
        )


def loaded_models(config: Config) -> set[str]:
    response = transport.ok(
        transport.request("GET", config.ollama_url + "/api/ps"), "Ollama loaded models"
    )
    return model_names(response)


def load_model(config: Config, model: ModelSpec) -> None:
    if model not in config.models:
        raise AppError("Refusing to preload a model outside this project's active catalog.")
    require_model(config)
    transport.ok(
        transport.request(
            "POST",
            config.ollama_url + "/api/generate",
            timeout=model.timeout,
            payload={
                "model": model.model,
                "stream": False,
                "keep_alive": model.keep_alive,
                "options": {"num_ctx": model.context_window},
            },
        ),
        "Model loading",
    )


def unload_model(config: Config, model: str) -> None:
    if model not in config.model_routes.values():
        raise AppError("Refusing to unload a model outside this project's configuration.")
    if model not in loaded_models(config):
        return  # Do not send an empty generation request for an unloaded model.
    transport.ok(
        transport.request(
            "POST",
            config.ollama_url + "/api/generate",
            timeout=UNLOAD_TIMEOUT_SECONDS,
            payload={"model": model, "stream": False, "keep_alive": 0},
        ),
        "Model unloading",
    )
    wait_for(
        "the selected model to unload",
        lambda: model not in loaded_models(config),
        UNLOAD_TIMEOUT_SECONDS,
    )


def unload_models(config: Config) -> None:
    errors = []
    for model in dict.fromkeys(config.model_routes.values()):
        try:
            unload_model(config, model)
        except (httpx.RequestError, AppError):
            errors.append(model)
    if errors:
        raise AppError("Could not unload configured models: " + ", ".join(errors))
