"""Tests for configurable CORS origins."""


def test_cors_parses_comma_separated_origins(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_CORS_ORIGINS", "https://a.com,https://b.com")
    from litellm_llmrouter.gateway.app import _parse_cors_origins

    origins = _parse_cors_origins()
    assert origins == ["https://a.com", "https://b.com"]


def test_cors_default_wildcard(monkeypatch):
    monkeypatch.delenv("ROUTEIQ_CORS_ORIGINS", raising=False)
    from litellm_llmrouter.gateway.app import _parse_cors_origins

    origins = _parse_cors_origins()
    assert origins == ["*"]


def test_cors_credentials_flag(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_CORS_CREDENTIALS", "true")
    from litellm_llmrouter.gateway.app import _parse_cors_credentials

    assert _parse_cors_credentials() is True


def test_cors_credentials_default(monkeypatch):
    monkeypatch.delenv("ROUTEIQ_CORS_CREDENTIALS", raising=False)
    from litellm_llmrouter.gateway.app import _parse_cors_credentials

    assert _parse_cors_credentials() is False
