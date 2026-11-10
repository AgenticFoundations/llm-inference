"""Configuration is data: dotenv files are never sourced by a shell."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import quote

from dotenv import dotenv_values

from .errors import AppError
from .models import INGRESS_TIMEOUT_SECONDS as INGRESS_TIMEOUT_SECONDS
from .models import ModelCatalog, ModelSpec
from .storage import private_json

SECRET_KEYS = ("LITELLM_MASTER_KEY", "LITELLM_SALT_KEY", "POSTGRES_PASSWORD")


@dataclass
class Config:
    root: Path
    values: dict[str, str] = field(repr=False)
    catalog: ModelCatalog

    @classmethod
    def load(cls, root: Path) -> "Config":
        if not (root / ".env").exists():
            raise AppError("Run make setup to create .env first.")
        defaults = dotenv_values(root / ".env.example", interpolate=False)
        supplied = dotenv_values(root / ".env", interpolate=False)
        values = {
            key: os.environ.get(key, supplied.get(key, default)) or ""
            for key, default in defaults.items()
        }
        if not values["MODELS_CONFIG"]:
            raise AppError("MODELS_CONFIG must point to a model catalog JSON file.")
        config = cls(root, values, ModelCatalog.load(root / values["MODELS_CONFIG"]))
        config.validate()
        return config

    def __getitem__(self, key: str) -> str:
        return self.values[key]

    def number(self, key: str) -> int:
        return int(self[key])

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        return self.catalog.chat_models

    @property
    def model_routes(self) -> dict[str, str]:
        return {model.alias: model.model for model in self.models}

    def snapshot(self) -> dict[str, object]:
        return {"values": dict(self.values), "catalog": self.catalog.to_dict()}

    @classmethod
    def from_snapshot(cls, root: Path, data: dict[str, object]) -> "Config":
        if "catalog" not in data:
            # Compatibility is limited to old saved sessions, so cleanup remains possible.
            values = cast(dict[str, str], data)
            try:
                catalog = ModelCatalog.from_legacy_session(values)
            except (KeyError, ValueError, TypeError) as exc:
                raise AppError("Cannot restore the saved model configuration.") from exc
        else:
            supplied = data.get("values")
            if not isinstance(supplied, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in supplied.items()
            ):
                raise AppError("Saved service configuration is malformed.")
            values = cast(dict[str, str], supplied)
            catalog = ModelCatalog.from_data(data["catalog"])
        config = cls(root, values, catalog)
        config.validate()
        return config

    def copilot_models(self) -> list[dict[str, object]]:
        return [
            {
                "id": model.alias,
                "name": model.name,
                "url": self.edge_url + "/v1/chat/completions",
                "apiType": "chat-completions",
                "contextWindow": model.context_window,
                "maxInputTokens": model.max_input_tokens,
                "maxOutputTokens": model.max_output_tokens,
                "toolCalling": model.capabilities.tools,
                "thinking": model.capabilities.thinking,
                "vision": "image" in model.capabilities.input,
            }
            for model in self.models
        ]

    @property
    def ollama_url(self) -> str:
        return f"http://127.0.0.1:{self['OLLAMA_PORT']}"

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self['GATEWAY_PORT']}"

    @property
    def edge_url(self) -> str:
        return f"http://127.0.0.1:{self['EDGE_PORT']}"

    def validate(self) -> None:
        for key, value in self.values.items():
            if any(char in value for char in "\r\n\x00"):
                raise AppError(f"{key} must be a single line.")
        ports = []
        for key in ("OLLAMA_PORT", "GATEWAY_PORT", "EDGE_PORT"):
            if not self[key].isdigit() or not 1024 <= int(self[key]) <= 65535:
                raise AppError(f"{key} must be a port between 1024 and 65535.")
            ports.append(int(self[key]))
        if len(set(ports)) != len(ports):
            raise AppError("Service ports must differ.")
        if not self["STARTUP_TIMEOUT"].isdigit() or int(self["STARTUP_TIMEOUT"]) < 1:
            raise AppError("STARTUP_TIMEOUT must be a positive integer.")
        for key in SECRET_KEYS:
            if len(self[key]) < 24:
                raise AppError(f"{key} must contain at least 24 characters; run make setup.")
        if not self["LITELLM_MASTER_KEY"].startswith("sk-"):
            raise AppError("LITELLM_MASTER_KEY must start with sk-.")
        for key in ("LITELLM_IMAGE", "POSTGRES_IMAGE", "NGINX_IMAGE"):
            image = self[key]
            tag = image.rsplit("/", 1)[-1]
            if not image or any(char.isspace() for char in image) or ":" not in tag:
                raise AppError(f"{key} must specify an explicit image tag or digest.")
            if tag.rsplit(":", 1)[-1] in ("latest", "main-latest", "main-stable", "lts"):
                raise AppError(f"{key} must use a version or digest rather than a moving tag.")

    def compose_env(self) -> dict[str, str]:
        keys = (
            "LITELLM_IMAGE",
            "POSTGRES_IMAGE",
            "NGINX_IMAGE",
            "GATEWAY_PORT",
            "EDGE_PORT",
            *SECRET_KEYS,
        )
        return {
            **{key: self[key] for key in keys},
            "DATABASE_URL": "postgresql://litellm:"
            + quote(self["POSTGRES_PASSWORD"], safe="")
            + "@db:5432/litellm",
        }

    def render(self) -> None:
        # JSON is valid YAML; this avoids templating/escaping model names into YAML.
        private_json(
            self.root / ".runtime/litellm.yaml",
            {
                "model_list": [
                    {
                        "model_name": model.alias,
                        "litellm_params": {
                            "model": "ollama_chat/" + model.model,
                            "api_base": f"http://host.docker.internal:{self['OLLAMA_PORT']}",
                            "keep_alive": model.keep_alive,
                            "num_ctx": model.context_window,
                            "timeout": model.timeout,
                            "max_tokens": model.max_output_tokens,
                        },
                        "model_info": {
                            "mode": "chat",
                            "max_input_tokens": model.max_input_tokens,
                            "max_output_tokens": model.max_output_tokens,
                        },
                    }
                    for model in self.models
                ],
                "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
                "litellm_settings": {
                    "telemetry": False,
                    "turn_off_message_logging": True,
                    "request_timeout": max(model.timeout for model in self.models),
                    "num_retries": 0,
                },
            },
        )
