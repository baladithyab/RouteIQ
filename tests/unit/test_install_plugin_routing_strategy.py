"""Regression guard for ``install_plugin_routing_strategy(app)`` — RouteIQ-6509.

THE historical bug (commits 59f80e9 + 7844419, called out in CLAUDE.md's
"Non-Obvious Behaviors"): ``install_plugin_routing_strategy`` MUST receive the
``app`` argument. If the arg is missing the resulting ``TypeError`` is silently
swallowed by the caller's broad ``except Exception`` and ML routing falls back to
LiteLLM's default WITHOUT any error — a silent, hard-to-spot regression.

These tests are TEST-ONLY (they do not modify the install function). They assert:

  1. The function signature REQUIRES ``app`` — calling it with no args raises
     ``TypeError`` (the swallowed-error trap the historical bug rode on).
  2. With ``use_plugin_strategy`` active + a real ``llm_router``, the install
     delegates to ``_install_plugin_strategy(llm_router, strategy_name)`` and
     returns its success — i.e. it WIRES the custom strategy, not the default.
  3. When the plugin strategy is NOT active it returns False (no install).
  4. The live call site in ``gateway/app.py`` PASSES ``app`` — proving the
     mechanism has a real caller that supplies the required arg (no
     mechanism-without-callsite, no silent-fallback regression).

All credential-free: ``llm_router`` and ``_install_plugin_strategy`` are
patched; no LiteLLM Router, no network, no AWS.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.startup import install_plugin_routing_strategy


def _app(*, use_plugin: bool = True) -> SimpleNamespace:
    """A minimal stand-in for the FastAPI app carrying ``state.use_plugin_strategy``
    (the only attribute the install reads)."""
    return SimpleNamespace(state=SimpleNamespace(use_plugin_strategy=use_plugin))


@pytest.fixture
def fake_llm_router():
    """Inject a stub ``litellm.proxy.proxy_server`` module exposing ``llm_router``.

    The real module cannot import in this credential-free env (it pulls boto3),
    and ``install_plugin_routing_strategy`` does ``from litellm.proxy.proxy_server
    import llm_router`` at call time — so we stand the module in via sys.modules.
    Yields the router MagicMock the test can configure; the test sets it to None
    to simulate an un-initialised proxy.
    """
    router = MagicMock()
    router.router_settings = {"routing_strategy": "llmrouter-knn"}
    stub = types.ModuleType("litellm.proxy.proxy_server")
    stub.llm_router = router
    saved = sys.modules.get("litellm.proxy.proxy_server")
    sys.modules["litellm.proxy.proxy_server"] = stub
    try:
        yield stub
    finally:
        if saved is not None:
            sys.modules["litellm.proxy.proxy_server"] = saved
        else:
            sys.modules.pop("litellm.proxy.proxy_server", None)


# --- (1) the ``app`` arg is REQUIRED (the swallowed-TypeError trap) --------


def test_app_argument_is_required_positional():
    """The signature has exactly one required positional param named ``app`` with
    no default — so a call missing it is a hard TypeError, not a silent no-op."""
    sig = inspect.signature(install_plugin_routing_strategy)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["app"]
    assert params[0].default is inspect.Parameter.empty


def test_calling_without_app_raises_type_error():
    """Calling with NO args raises TypeError (the historical silent-fallback bug
    rode on this being swallowed upstream — here we assert it is raised)."""
    with pytest.raises(TypeError):
        install_plugin_routing_strategy()  # type: ignore[call-arg]


# --- (2) it WIRES the custom strategy (not the silent default) -------------


def test_install_wires_custom_strategy_via_install_plugin_strategy(fake_llm_router):
    """Plugin strategy active + a real llm_router -> the install calls
    _install_plugin_strategy(llm_router, strategy_name) and returns its result.
    This is the positive path the regression would silently skip."""
    with patch(
        "litellm_llmrouter.gateway.app._install_plugin_strategy",
        return_value=True,
    ) as mock_install:
        result = install_plugin_routing_strategy(_app(use_plugin=True))

    assert result is True
    # the custom strategy was wired onto the live router (not the LiteLLM default).
    mock_install.assert_called_once()
    called_router, called_strategy = mock_install.call_args[0][:2]
    assert called_router is fake_llm_router.llm_router
    assert called_strategy == "llmrouter-knn"


def test_install_returns_install_plugin_strategy_failure(fake_llm_router):
    """When _install_plugin_strategy reports failure, the install returns False
    (so the caller can log the fall-back — not swallow it)."""
    fake_llm_router.llm_router.router_settings = {}
    with patch(
        "litellm_llmrouter.gateway.app._install_plugin_strategy",
        return_value=False,
    ):
        assert install_plugin_routing_strategy(_app(use_plugin=True)) is False


def test_install_noop_when_plugin_strategy_inactive():
    """When app.state.use_plugin_strategy is False the install is a no-op
    (returns False) and never touches the router."""
    with patch(
        "litellm_llmrouter.gateway.app._install_plugin_strategy"
    ) as mock_install:
        assert install_plugin_routing_strategy(_app(use_plugin=False)) is False
    mock_install.assert_not_called()


def test_install_returns_false_when_router_not_initialised(fake_llm_router):
    """A None llm_router (proxy not initialised) -> False, no install attempt."""
    fake_llm_router.llm_router = None
    with patch(
        "litellm_llmrouter.gateway.app._install_plugin_strategy"
    ) as mock_install:
        assert install_plugin_routing_strategy(_app(use_plugin=True)) is False
    mock_install.assert_not_called()


def test_explicit_env_override_strategy_name(monkeypatch, fake_llm_router):
    """ROUTEIQ_ROUTING_STRATEGY (env) overrides the router-config strategy."""
    monkeypatch.setenv("ROUTEIQ_ROUTING_STRATEGY", "llmrouter-svm")
    with patch(
        "litellm_llmrouter.gateway.app._install_plugin_strategy",
        return_value=True,
    ) as mock_install:
        install_plugin_routing_strategy(_app(use_plugin=True))

    # env override wins over the router-config value.
    assert mock_install.call_args[0][1] == "llmrouter-svm"


# --- (4) the live call site PASSES app (no silent-fallback regression) -----


def test_live_call_site_passes_app_argument():
    """gateway/app.py must call install_plugin_routing_strategy(app) WITH app —
    grep-prove the real caller supplies the required arg so the swallowed-error
    regression can't silently disable ML routing."""
    app_py = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "litellm_llmrouter"
        / "gateway"
        / "app.py"
    )
    source = app_py.read_text()
    # the caller passes the app positional, not a bare no-arg call.
    assert "install_plugin_routing_strategy(app)" in source
    assert "install_plugin_routing_strategy()" not in source
