import subprocess

import pytest

from llm_inference import runtime
from llm_inference.config import AppError


@pytest.mark.parametrize(
    "failure,expected",
    [
        (FileNotFoundError("private details"), "Required executable not found: docker"),
        (
            subprocess.TimeoutExpired(["docker", "private argument"], 3),
            "Command timed out: docker",
        ),
        (
            subprocess.CompletedProcess(
                ["docker", "private argument"], 7, "private output", "private error"
            ),
            "docker command failed (exit 7)",
        ),
    ],
)
def test_subprocess_failures_are_actionable_and_do_not_leak_details(monkeypatch, failure, expected):
    def fake_run(*_args, **_kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    with pytest.raises(AppError) as exc:
        runtime.run(["docker", "private argument"], env={"PRIVATE_SECRET": "private value"})
    assert expected in str(exc.value)
    assert "private" not in str(exc.value)


def test_compose_keeps_resolved_binary_when_environment_changes(config, monkeypatch, tmp_path):
    binary = str(tmp_path / "original" / "docker")
    monkeypatch.setattr(runtime, "docker_binary", lambda: binary)
    compose = runtime.Compose(config)

    def changed_installation():
        raise AssertionError("The Docker binary should only be resolved once per Compose instance")

    monkeypatch.setattr(runtime, "docker_binary", changed_installation)
    assert compose.command[0] == binary
    assert compose.env["PATH"].startswith(str(tmp_path / "original"))
    calls = []
    monkeypatch.setattr(runtime, "run", lambda args, **_kwargs: calls.append(args) or "")
    compose.preflight()
    assert calls[0][0] == calls[1][0] == binary
