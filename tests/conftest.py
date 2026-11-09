import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values, set_key

from llm_inference.bootstrap import setup
from llm_inference.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path, monkeypatch):
    shutil.copy(ROOT / ".env.example", tmp_path / ".env.example")
    for key in dotenv_values(tmp_path / ".env.example", interpolate=False):
        monkeypatch.delenv(key, raising=False)
    setup(tmp_path)
    # Keep protocol tests independent of the production model's context default.
    set_key(tmp_path / ".env", "OLLAMA_CONTEXT_LENGTH", "16384", quote_mode="never")
    return Config.load(tmp_path)
