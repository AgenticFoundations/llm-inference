import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from dotenv import dotenv_values

from llm_inference.bootstrap import setup
from llm_inference.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_files(tmp_path, monkeypatch):
    shutil.copy(ROOT / ".env.example", tmp_path / ".env.example")
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/models.json", tmp_path / "config/models.json")
    for key in dotenv_values(tmp_path / ".env.example", interpolate=False):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def config(project_files):
    setup(project_files)
    # Keep protocol tests independent of the production model's context default.
    path = project_files / "config/models.json"
    data = json.loads(path.read_text())
    model = data["models"][0]
    model.update(context_window=16384, max_input_tokens=14336, max_output_tokens=2048)
    path.write_text(json.dumps({"models": [model]}))
    return Config.load(project_files)


@pytest.fixture
def second_model(config):
    return replace(
        config.models[0],
        alias="local-coder-bf16",
        name="Local Qwen BF16",
        model="qwen3.8:27b-mlx-bf16",
        context_window=8192,
        max_input_tokens=7168,
        max_output_tokens=1024,
        keep_alive="2m",
        timeout=120,
    )
