import re
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from llm_inference.errors import AppError
from llm_inference.models import INGRESS_TIMEOUT_SECONDS


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
    from llm_inference.models import ModelCatalog

    config.catalog = ModelCatalog((replace(config.models[0], timeout=INGRESS_TIMEOUT_SECONDS),))
    config.validate()
    with pytest.raises(AppError, match="timeout"):
        replace(config.models[0], timeout=INGRESS_TIMEOUT_SECONDS + 1)
