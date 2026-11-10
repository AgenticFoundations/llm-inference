"""Gateway key management and checks. Error bodies never reach logs."""

import json
import re
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import httpx

from . import transport
from .config import Config
from .errors import AppError
from .models import ModelSpec
from .storage import private_json


class StoredKey(TypedDict):
    key: str
    model: NotRequired[str]
    models: NotRequired[list[str]]


def key_path(config: Config, name: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise AppError("Client name must be 1–64 letters, numbers, underscores, or hyphens.")
    return config.root / ".runtime/keys" / f"{name}.json"


def client_key(config: Config, name: str = "copilot") -> str:
    path = key_path(config, name)
    aliases = list(config.model_routes)
    if path.exists():
        stored = cast(StoredKey, json.loads(path.read_text()))
        response = transport.request("GET", config.gateway_url + "/v1/models", key=stored["key"])
        if response.is_success:
            previous = stored.get("models", [stored.get("model", "")])
            if set(previous) != set(aliases):
                # Keep the same raw key so existing clients' secure-store entries still work.
                transport.ok(
                    transport.request(
                        "POST",
                        config.gateway_url + "/key/update",
                        key=config["LITELLM_MASTER_KEY"],
                        payload={"key": stored["key"], "models": aliases},
                    ),
                    "Client-key model update",
                )
                private_json(path, {"key": stored["key"], "models": aliases})
            return stored["key"]
        if response.status_code not in (401, 403):
            transport.ok(response, "Client-key validation")
        revoke_key(config, name)
    response = transport.ok(
        transport.request(
            "POST",
            config.gateway_url + "/key/generate",
            key=config["LITELLM_MASTER_KEY"],
            payload={"models": aliases, "key_alias": name},
        ),
        "Client-key creation",
    )
    key = cast(str, response.json()["key"])
    private_json(path, {"key": key, "models": aliases})
    return key


def revoke_key(config: Config, name: str) -> None:
    path = key_path(config, name)
    if not path.exists():
        return
    key = json.loads(path.read_text())["key"]
    transport.ok(
        transport.request(
            "POST",
            config.gateway_url + "/key/delete",
            key=config["LITELLM_MASTER_KEY"],
            payload={"keys": [key]},
        ),
        "Client-key revocation",
    )
    path.unlink()


def check_routes(config: Config, key: str, base_url: str | None = None) -> None:
    base = base_url or config.edge_url
    for credential in (None, "sk-deliberately-invalid"):
        response = transport.request("GET", base + "/v1/models", key=credential)
        if response.status_code not in (401, 403):
            raise AppError("Gateway did not reject an absent/invalid API key.")
    response = transport.ok(
        transport.request("GET", base + "/v1/models", key=key), "Model discovery"
    )
    if not set(config.model_routes) <= {model["id"] for model in response.json()["data"]}:
        raise AppError("A configured model alias is absent from gateway discovery.")
    for path in ("/ui", "/key/generate", "/config/yaml", "/model/new", "/api/tags"):
        response = transport.request("POST", base + path, key=key, payload={})
        if response.status_code != 404:
            raise AppError(f"Ingress unexpectedly exposes {path}.")
    for method, path in (("POST", "/v1/models"), ("GET", "/v1/chat/completions")):
        if transport.request(method, base + path, key=key).status_code != 405:
            raise AppError("Ingress did not reject an unsupported HTTP method.")


def check_inference(
    config: Config, key: str, model: ModelSpec, base_url: str | None = None
) -> None:
    body = {
        "model": model.alias,
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "max_tokens": min(32, model.max_output_tokens),
    }
    url = (base_url or config.edge_url) + "/v1/chat/completions"
    response = transport.ok(
        transport.request("POST", url, key=key, payload=body, timeout=model.timeout),
        "Inference",
    )
    try:
        data = response.json()
    except ValueError:
        raise AppError("Inference returned invalid JSON.") from None
    if not isinstance(data, dict) or not data.get("choices"):
        raise AppError("Inference returned no choices.")
    # Consume SSE incrementally; a 200 alone does not prove a working stream.
    saw_choice = saw_done = False
    with httpx.stream(
        "POST",
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={**body, "stream": True},
        timeout=model.timeout,
        trust_env=False,
    ) as stream:
        transport.ok(stream, "Streaming inference")
        for line in stream.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                raise AppError("Gateway returned invalid JSON in the inference stream.") from None
            if not isinstance(event, dict):
                raise AppError("Gateway returned a non-object event in the inference stream.")
            if "error" in event:
                raise AppError("Gateway returned an error inside the inference stream.")
            saw_choice |= bool(event.get("choices"))
    if not (saw_choice and saw_done):
        raise AppError("Inference stream ended without choices and a completion marker.")
