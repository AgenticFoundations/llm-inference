"""Validated model catalog shared by gateway routing, lifecycle, and client exports."""

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import cast

from .errors import AppError

# Nginx's proxy timeout is an inactivity limit, not a total response deadline.
INGRESS_TIMEOUT_SECONDS = 900


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AppError(f"{label} must be an object.")
    if set(value) - fields:
        raise AppError(
            f"{label} contains unknown settings: {', '.join(sorted(set(value) - fields))}."
        )
    return cast(dict[str, object], value)


def _text(data: dict[str, object], key: str, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\x00"):
        raise AppError(f"Model setting {key} must be a nonempty single-line string.")
    return value


def _number(data: dict[str, object], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if type(value) is not int or value < 1:
        raise AppError(f"Model setting {key} must be a positive integer.")
    return value


@dataclass(frozen=True)
class Capabilities:
    input: tuple[str, ...]
    output: tuple[str, ...]
    tools: bool = False
    thinking: bool = False

    @classmethod
    def from_data(cls, value: object) -> "Capabilities":
        data = _object(value, {"input", "output", "tools", "thinking"}, "Model capabilities")
        inputs, outputs = data.get("input"), data.get("output")
        if (
            not isinstance(inputs, list)
            or not inputs
            or any(item not in ("text", "image") for item in inputs)
        ):
            raise AppError("Chat input modalities must be a nonempty list of text and/or image.")
        if "text" not in inputs or outputs != ["text"]:
            raise AppError(
                "Chat models must accept text and produce text; "
                "image generation needs a separate backend."
            )
        tools, thinking = data.get("tools", False), data.get("thinking", False)
        if not isinstance(tools, bool) or not isinstance(thinking, bool):
            raise AppError("Model tools and thinking capabilities must be true or false.")
        return cls(tuple(inputs), ("text",), tools, thinking)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "input": list(self.input), "output": list(self.output)}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    name: str
    model: str
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    capabilities: Capabilities
    keep_alive: str = "5m"
    timeout: int = INGRESS_TIMEOUT_SECONDS
    enabled: bool = True
    type: str = field(default="chat", init=False)

    @classmethod
    def from_data(cls, value: object) -> "ModelSpec":
        data = _object(value, {item.name for item in fields(cls)}, "Model")
        if data.get("type", "chat") != "chat":
            raise AppError("Only chat models are currently supported by this gateway.")
        alias = _text(data, "alias")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise AppError("Model enabled must be true or false.")
        return cls(
            alias,
            _text(data, "name", alias),
            _text(data, "model"),
            _number(data, "context_window"),
            _number(data, "max_input_tokens"),
            _number(data, "max_output_tokens"),
            Capabilities.from_data(data.get("capabilities")),
            _text(data, "keep_alive", "5m"),
            _number(data, "timeout", INGRESS_TIMEOUT_SECONDS),
            enabled,
        )

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.alias):
            raise AppError(
                "Model aliases must use letters, numbers, dots, underscores, or hyphens."
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", self.model):
            raise AppError(f"Model {self.alias} must use an exact installed Ollama tag.")
        for key in ("context_window", "max_input_tokens", "max_output_tokens", "timeout"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise AppError(f"Model {self.alias}: {key} must be a positive integer.")
        if self.max_input_tokens + self.max_output_tokens > self.context_window:
            raise AppError(f"Model {self.alias}: input and output budgets exceed context_window.")
        if self.timeout > INGRESS_TIMEOUT_SECONDS:
            raise AppError(
                f"Model {self.alias}: timeout cannot exceed {INGRESS_TIMEOUT_SECONDS} seconds."
            )
        if not re.fullmatch(r"(?:0|[1-9]\d*(?:ms|s|m|h))", self.keep_alive):
            raise AppError(
                f"Model {self.alias}: keep_alive must be 0 or a finite duration such as 5m."
            )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "capabilities": self.capabilities.to_dict()}


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[ModelSpec, ...]
    # Image generation has no backend yet. Preserve its inactive JSON without an adapter.
    images: tuple[dict[str, object], ...] = ()

    @property
    def chat_models(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.models if model.enabled)

    @classmethod
    def load(cls, path: Path) -> "ModelCatalog":
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise AppError(f"Cannot read model catalog {path}; expected valid JSON.") from exc
        return cls.from_data(data)

    @classmethod
    def from_data(cls, value: object) -> "ModelCatalog":
        data = _object(value, {"models"}, "Model catalog")
        entries = data.get("models")
        if not isinstance(entries, list) or not entries:
            raise AppError("Model catalog must contain a nonempty models list.")
        models: list[ModelSpec] = []
        images: list[dict[str, object]] = []
        for item in entries:
            if isinstance(item, dict) and item.get("type") == "image_generation":
                image = _object(
                    item,
                    {"alias", "name", "type", "enabled", "model", "endpoint", "parameters"},
                    "Image-generation entry",
                )
                if image.get("enabled", False) is not False:
                    raise AppError("Image generation is catalog-only; leave enabled false.")
                alias = _text(image, "alias")
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", alias):
                    raise AppError(
                        "Invalid image-generation alias; use letters, numbers, '.', '_' or '-'."
                    )
                if not isinstance(image.get("parameters", {}), dict):
                    raise AppError("Image-generation parameters must be an object.")
                images.append({**image, "enabled": False})
            else:
                models.append(ModelSpec.from_data(item))
        return cls(tuple(models), tuple(images))

    def __post_init__(self) -> None:
        aliases = [model.alias for model in self.models] + [
            str(image["alias"]) for image in self.images
        ]
        if not aliases or len(set(aliases)) != len(aliases):
            raise AppError("Model aliases must be unique and the catalog must not be empty.")
        if not self.chat_models:
            raise AppError("At least one chat model must be enabled.")

    def get(self, alias: str) -> ModelSpec:
        for model in self.chat_models:
            if model.alias == alias:
                return model
        raise AppError(f"Unknown or disabled chat model alias: {alias}.")

    def to_dict(self) -> dict[str, object]:
        return {"models": [model.to_dict() for model in self.models] + list(self.images)}

    @classmethod
    def from_legacy_session(cls, values: dict[str, str]) -> "ModelCatalog":
        """Read old saved sessions for safe cleanup; never used to load current .env models."""
        routes = {
            values["MODEL_ALIAS"]: values["OLLAMA_MODEL"],
            **json.loads(values.get("OLLAMA_ADDITIONAL_MODELS", "{}")),
        }
        context = int(values["OLLAMA_CONTEXT_LENGTH"])
        output = min(32768, max(1, context // 8))
        return cls(
            tuple(
                ModelSpec(
                    alias=alias,
                    name=alias,
                    model=tag,
                    context_window=context,
                    max_input_tokens=context - output,
                    max_output_tokens=output,
                    capabilities=Capabilities(
                        ("text", "image"), ("text",), tools=True, thinking=True
                    ),
                    keep_alive=values["OLLAMA_KEEP_ALIVE"],
                    timeout=int(values["REQUEST_TIMEOUT"]),
                )
                for alias, tag in routes.items()
            )
        )
