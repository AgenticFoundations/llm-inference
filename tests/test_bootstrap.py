import shutil
import stat
from pathlib import Path

from dotenv import dotenv_values

from llm_inference.bootstrap import setup
from llm_inference.config import SECRET_KEYS, Config


def test_setup_generates_distinct_private_credentials(project_files, monkeypatch, capsys):
    tmp_path = project_files
    template = Path(__file__).resolve().parents[1] / ".env.example"
    shutil.copy(template, tmp_path / ".env.example")
    for key in dotenv_values(template, interpolate=False):
        monkeypatch.delenv(key, raising=False)
    setup(tmp_path)
    config = Config.load(tmp_path)
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600
    assert len({config[key] for key in SECRET_KEYS}) == len(SECRET_KEYS)
    output = capsys.readouterr().out
    for key in SECRET_KEYS:
        value = config[key]
        assert len(value) >= 64
        if key.startswith("LITELLM"):
            assert value.startswith("sk-")
        assert value not in output
    assert (tmp_path / ".env.example").read_bytes() == template.read_bytes()


def test_setup_preserves_existing_credentials(config):
    path = config.root / ".env"
    before = path.read_bytes()
    setup(config.root)
    assert path.read_bytes() == before
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
