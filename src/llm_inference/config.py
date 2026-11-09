"""Configuration is data: dotenv files are never sourced by a shell."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from dotenv import dotenv_values

from .storage import private_json

SECRET_KEYS = ("LITELLM_MASTER_KEY", "LITELLM_SALT_KEY", "POSTGRES_PASSWORD")
# Keep aligned with Nginx's proxy timeouts; these measure inactivity, not total duration.
INGRESS_TIMEOUT_SECONDS = 900


class AppError(Exception):
    """An actionable error suitable for the terminal."""


@dataclass
class Config:
    root: Path
    values: dict[str, str] = field(repr=False)

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
        config = cls(root, values)
        config.validate()
        return config

    def __getitem__(self, key: str) -> str:
        return self.values[key]

    def number(self, key: str) -> int:
        return int(self[key])

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
        for key in (
            "STARTUP_TIMEOUT",
            "REQUEST_TIMEOUT",
            "OLLAMA_CONTEXT_LENGTH",
        ):
            if not self[key].isdigit() or int(self[key]) < 1:
                raise AppError(f"{key} must be a positive integer.")
        if self.number("REQUEST_TIMEOUT") > INGRESS_TIMEOUT_SECONDS:
            raise AppError(
                f"REQUEST_TIMEOUT cannot exceed the ingress timeout of {INGRESS_TIMEOUT_SECONDS} "
                "seconds."
            )
        if not re.fullmatch(r"(?:0|[1-9]\d*(?:ms|s|m|h))", self["OLLAMA_KEEP_ALIVE"]):
            raise AppError("OLLAMA_KEEP_ALIVE must be 0 or a finite duration such as 5m.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", self["OLLAMA_MODEL"]):
            raise AppError("OLLAMA_MODEL must be an exact local Ollama model tag.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self["MODEL_ALIAS"]):
            raise AppError("MODEL_ALIAS must use letters, numbers, dots, underscores, or hyphens.")
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
                        "model_name": self["MODEL_ALIAS"],
                        "litellm_params": {
                            "model": "ollama_chat/" + self["OLLAMA_MODEL"],
                            "api_base": f"http://host.docker.internal:{self['OLLAMA_PORT']}",
                            "keep_alive": self["OLLAMA_KEEP_ALIVE"],
                            "num_ctx": self.number("OLLAMA_CONTEXT_LENGTH"),
                            "timeout": self.number("REQUEST_TIMEOUT"),
                        },
                    }
                ],
                "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
                "litellm_settings": {
                    "telemetry": False,
                    "turn_off_message_logging": True,
                    "request_timeout": self.number("REQUEST_TIMEOUT"),
                    "num_retries": 0,
                },
            },
        )
