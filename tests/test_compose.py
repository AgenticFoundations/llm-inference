import re
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values

from llm_inference.config import INGRESS_TIMEOUT_SECONDS, AppError


def test_container_exposure_and_manual_lifecycle_contract():
    compose = yaml.safe_load((Path(__file__).resolve().parents[1] / "compose.yaml").read_text())
    services = compose["services"]
    for service in services.values():
        assert service["restart"] == "no"
        for binding in service.get("ports", []):
            assert binding.startswith("127.0.0.1:")
        assert not any("docker.sock" in mount for mount in service.get("volumes", []))
    assert "ports" not in services["db"]
    assert "volumes" not in services["gateway"]
    assert "volumes" not in services["edge"]
    assert set(services) == {"db", "gateway", "edge"}
    for service in services.values():
        assert set(service["networks"]) <= set(compose["networks"])
    assert not any(network and network.get("external") for network in compose["networks"].values())
    assert compose["networks"]["database"]["internal"] is True


def test_nginx_matches_request_timeout_cap(config):
    root = Path(__file__).resolve().parents[1]
    nginx = (root / "config/nginx.conf").read_text()
    for directive in ("proxy_read_timeout", "proxy_send_timeout"):
        match = re.search(rf"^\s*{directive}\s+(\d+)s;", nginx, re.MULTILINE)
        assert match is not None
        assert int(match[1]) == INGRESS_TIMEOUT_SECONDS
    defaults = dotenv_values(root / ".env.example", interpolate=False)
    assert int(defaults["REQUEST_TIMEOUT"]) <= INGRESS_TIMEOUT_SECONDS
    config.values["REQUEST_TIMEOUT"] = str(INGRESS_TIMEOUT_SECONDS)
    config.validate()
    config.values["REQUEST_TIMEOUT"] = str(INGRESS_TIMEOUT_SECONDS + 1)
    with pytest.raises(AppError, match="REQUEST_TIMEOUT"):
        config.validate()
