"""Control model residency through the user's existing Ollama server."""

import httpx

from . import transport
from .config import AppError, Config
from .runtime import wait_for

UNLOAD_TIMEOUT_SECONDS = 30


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
    names = {item["name"] for item in response.json()["models"]}
    if config["OLLAMA_MODEL"] not in names:
        available = ", ".join(sorted(names)) or "none"
        raise AppError(
            f"Model {config['OLLAMA_MODEL']} is not installed. Available: {available}. "
            "Set OLLAMA_MODEL to the exact installed tag. No automatic downloads."
        )


def loaded_models(config: Config) -> set[str]:
    response = transport.ok(
        transport.request("GET", config.ollama_url + "/api/ps"), "Ollama loaded models"
    )
    return {str(item["name"]) for item in response.json()["models"]}


def load_model(config: Config) -> None:
    require_model(config)
    transport.ok(
        transport.request(
            "POST",
            config.ollama_url + "/api/generate",
            timeout=config.number("REQUEST_TIMEOUT"),
            payload={
                "model": config["OLLAMA_MODEL"],
                "stream": False,
                "keep_alive": config["OLLAMA_KEEP_ALIVE"],
                "options": {"num_ctx": config.number("OLLAMA_CONTEXT_LENGTH")},
            },
        ),
        "Model loading",
    )


def unload_model(config: Config) -> None:
    model = config["OLLAMA_MODEL"]
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
