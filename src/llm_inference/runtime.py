"""Project Compose commands, subprocesses, lifecycle locking, and readiness helpers."""

import fcntl
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .config import Config
from .errors import AppError

ORB_DOCKER = Path("/Applications/OrbStack.app/Contents/MacOS/xbin/docker")
PROJECT = "llm-inference"


def docker_binary() -> str:
    if ORB_DOCKER.is_file():
        return str(ORB_DOCKER)
    binary = shutil.which("docker")
    if binary:
        return binary
    raise AppError("OrbStack's Docker CLI was not found. Install/open OrbStack first.")


def run(
    args: list[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = 30,
    capture: bool = True,
) -> str:
    try:
        result = subprocess.run(
            args,
            env=env,
            timeout=timeout,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise AppError(f"Required executable not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AppError(f"Command timed out: {Path(args[0]).name}") from exc
    if result.returncode:
        # Never echo environment variables, full command lines, or provider error bodies.
        raise AppError(
            f"{Path(args[0]).name} command failed (exit {result.returncode}). "
            "Check OrbStack is running; use make logs for service errors."
        )
    return result.stdout or ""


class Compose:
    def __init__(self, config: Config):
        self.config = config
        self.binary = docker_binary()
        standalone = Path(self.binary).with_name("docker-compose")
        self._prefix = (
            [str(standalone)]
            if standalone.is_file()
            else [self.binary, "--context", "orbstack", "compose"]
        )

    @property
    def command(self) -> list[str]:
        return self._prefix + [
            "--project-name",
            PROJECT,
            "--project-directory",
            str(self.config.root),
            "--env-file",
            "/dev/null",
            "-f",
            str(self.config.root / "compose.yaml"),
        ]

    @property
    def env(self) -> dict[str, str]:
        values = dict(os.environ)
        for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "COMPOSE_FILE", "COMPOSE_PROFILES"):
            values.pop(key, None)
        values["DOCKER_CONTEXT"] = "orbstack"
        # Find OrbStack's bundled Compose plugin even when Docker isn't on PATH.
        values["PATH"] = str(Path(self.binary).parent) + os.pathsep + values.get("PATH", "")
        values.update(self.config.compose_env())
        return values

    def call(self, *args: str, timeout: float | None = 30, capture: bool = True) -> str:
        return run(self.command + list(args), env=self.env, timeout=timeout, capture=capture)

    def preflight(self) -> None:
        try:
            run([self.binary, "context", "inspect", "orbstack"], env=self.env)
        except AppError as exc:
            raise AppError(
                "OrbStack's Docker context is unavailable. Open OrbStack once to finish "
                "its setup, then run make doctor."
            ) from exc
        run([self.binary, "--context", "orbstack", "info"], env=self.env)
        self.call("version")
        self.call("config", "--quiet")


@contextmanager
def lifecycle_lock(root: Path) -> Iterator[None]:
    directory = root / ".runtime"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "lifecycle.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AppError(
                "Another lifecycle command is running. Try again when it finishes."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for(description: str, probe: Callable[[], bool], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if probe():
            return
        if time.monotonic() >= deadline:
            raise AppError(f"Timed out waiting for {description}.")
        time.sleep(0.5)
