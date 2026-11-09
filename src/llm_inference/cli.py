import argparse
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Never

import httpx

from . import gateway
from .bootstrap import setup
from .config import AppError, Config
from .controller import Controller
from .runtime import lifecycle_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="On-demand native Ollama + OrbStack gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "setup",
        "doctor",
        "up",
        "down",
        "restart",
        "status",
        "check",
        "model-start",
        "model-stop",
    ):
        sub.add_parser(name)
    logs = sub.add_parser("logs")
    logs.add_argument("service", nargs="?", choices=["gateway", "db", "edge"])
    for command in ("key", "revoke"):
        sub.add_parser(command).add_argument("name", nargs="?", default="copilot")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    controller = Controller(root)

    def dispatch() -> None:
        match args.command:
            case "setup":
                setup(root)
            case "restart":
                Config.load(root)  # Reject invalid edits before stopping a working session.
                controller.down()
                controller.up()
            case "logs":
                controller.logs(args.service)
            case "model-start":
                controller.model_start()
            case "model-stop":
                controller.model_stop()
            case "key":
                print(gateway.client_key(controller.active_config(), args.name))
            case "revoke":
                gateway.revoke_key(controller.active_config(), args.name)
                print(f"Revoked client: {args.name}")
            case _:
                getattr(controller, args.command)()

    def terminate(_signum: int, _frame: FrameType | None) -> Never:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        if args.command in (
            "setup",
            "up",
            "down",
            "restart",
            "key",
            "revoke",
            "model-start",
            "model-stop",
        ):
            with lifecycle_lock(root):
                dispatch()
        else:
            dispatch()
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except httpx.RequestError:
        print(
            "Error: Service connection failed. Inspect make status and make logs.", file=sys.stderr
        )
        return 1
    except (KeyError, ValueError):
        print(
            "Error: Invalid service response or runtime state. Inspect make logs.", file=sys.stderr
        )
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0
