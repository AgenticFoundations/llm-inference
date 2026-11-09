"""First-use configuration and credential generation; starts no services."""

import os
import secrets
from pathlib import Path

from .config import SECRET_KEYS, Config


def setup(root: Path) -> None:
    path = root / ".env"
    if path.exists():
        print(".env already exists; preserved existing credentials and configuration.")
        Config.load(root)
        return
    content = (root / ".env.example").read_text()
    for key in SECRET_KEYS:
        secret = ("sk-" if key.startswith("LITELLM") else "") + secrets.token_hex(32)
        content = content.replace(f"{key}=\n", f"{key}={secret}\n")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(content)
    print("Created private .env with generated credentials. No services started.")
