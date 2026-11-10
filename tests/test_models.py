import copy
import json

import pytest

from llm_inference.errors import AppError
from llm_inference.models import ModelCatalog


@pytest.mark.parametrize(
    "field,value",
    [
        ("alias", "../bad"),
        ("model", "invalid model"),
        ("model", "model\nother"),
        ("context_window", 0),
        ("context_window", True),
        ("context_window", 16383),
        ("max_output_tokens", -1),
        ("max_input_tokens", 0),
        ("keep_alive", "-1"),
        ("timeout", 901),
        ("enabled", "yes"),
        ("context_widnow", 16384),
    ],
)
def test_invalid_per_model_configuration_is_rejected(config, field, value):
    data = copy.deepcopy(config.catalog.to_dict())
    data["models"][0][field] = value
    with pytest.raises(AppError):
        ModelCatalog.from_data(data)


@pytest.mark.parametrize("kind", ["chat", "image_generation"])
def test_duplicate_aliases_are_rejected(config, kind):
    model = config.models[0].to_dict()
    duplicate = (
        model
        if kind == "chat"
        else {"alias": model["alias"], "type": "image_generation", "enabled": False}
    )
    with pytest.raises(AppError, match="unique"):
        ModelCatalog.from_data({"models": [model, duplicate]})


@pytest.mark.parametrize("value", [None, [], {}, {"models": []}, {"models": "wrong"}])
def test_catalog_requires_models(value):
    with pytest.raises(AppError):
        ModelCatalog.from_data(value)


def test_missing_or_invalid_json_catalog_has_actionable_error(tmp_path):
    path = tmp_path / "models.json"
    with pytest.raises(AppError, match="Cannot read model catalog"):
        ModelCatalog.load(path)
    path.write_text('{"invalid"')
    with pytest.raises(AppError, match="valid JSON"):
        ModelCatalog.load(path)


def test_disabled_image_generation_entry_is_saved_but_not_routed(config):
    image = {
        "alias": "local-image",
        "type": "image_generation",
        "enabled": False,
        "model": None,
        "endpoint": None,
        "parameters": {"width": 1024, "height": 1024},
    }
    data = config.catalog.to_dict()
    data["models"].append(image)
    catalog = ModelCatalog.from_data(data)
    assert len(catalog.chat_models) == 1
    assert catalog.images[0]["parameters"] == {"width": 1024, "height": 1024}
    assert ModelCatalog.from_data(json.loads(json.dumps(catalog.to_dict()))) == catalog
    config.catalog = catalog
    config.render()
    routes = json.loads((config.root / ".runtime/litellm.yaml").read_text())["model_list"]
    assert [route["model_name"] for route in routes] == ["local-coder"]
    assert [entry["id"] for entry in config.copilot_models()] == ["local-coder"]


def test_image_generation_cannot_be_enabled_without_integrating_backend(config):
    data = config.catalog.to_dict()
    data["models"].append({"alias": "local-image", "type": "image_generation", "enabled": True})
    with pytest.raises(AppError, match="catalog-only"):
        ModelCatalog.from_data(data)


def test_no_default_alias_is_needed_and_selection_is_explicit(config):
    assert set(config.catalog.to_dict()) == {"models"}
    assert config.catalog.get("local-coder") == config.models[0]
    with pytest.raises(AppError, match="Unknown or disabled"):
        config.catalog.get("missing")


def test_disabled_chat_model_is_not_an_active_route(config, second_model):
    data = config.catalog.to_dict()
    extra = second_model.to_dict()
    extra["enabled"] = False
    data["models"].append(extra)
    catalog = ModelCatalog.from_data(data)
    assert len(catalog.chat_models) == 1
    with pytest.raises(AppError, match="disabled"):
        catalog.get(second_model.alias)
