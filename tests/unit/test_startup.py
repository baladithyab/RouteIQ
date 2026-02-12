"""
Unit Tests for startup.py
==========================

Tests for the startup orchestration module:
- register_router_decision_callback (enabled/disabled/import-error)
- register_strategies (success/import-error)
- start_config_sync_if_enabled (enabled/disabled)
- init_observability_if_enabled (enabled/disabled/import-error)
- init_mcp_tracing_if_enabled (enabled/disabled)
- init_a2a_tracing_if_enabled (middleware registration, gateway instrumentation)
- main() (arg parsing, orchestration order)

All tests mock external dependencies (litellm, uvicorn, etc.).
"""

import os
from unittest.mock import MagicMock, patch


# =============================================================================
# register_router_decision_callback
# =============================================================================


class TestRegisterRouterDecisionCallback:
    def test_disabled_via_env(self):
        with patch.dict(os.environ, {"LLMROUTER_ROUTER_CALLBACK_ENABLED": "false"}):
            from litellm_llmrouter.startup import register_router_decision_callback

            result = register_router_decision_callback()
            assert result is None

    def test_enabled_success(self):
        with patch.dict(os.environ, {"LLMROUTER_ROUTER_CALLBACK_ENABLED": "true"}):
            mock_callback = MagicMock()
            with patch(
                "litellm_llmrouter.router_decision_callback.register_router_decision_callback",
                return_value=mock_callback,
            ):
                from litellm_llmrouter.startup import register_router_decision_callback

                result = register_router_decision_callback()
                assert result is mock_callback

    def test_import_error_returns_none(self):
        with patch.dict(os.environ, {"LLMROUTER_ROUTER_CALLBACK_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.router_decision_callback.register_router_decision_callback",
                side_effect=ImportError("no module"),
            ):
                from litellm_llmrouter.startup import register_router_decision_callback

                result = register_router_decision_callback()
                assert result is None

    def test_generic_exception_returns_none(self):
        with patch.dict(os.environ, {"LLMROUTER_ROUTER_CALLBACK_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.router_decision_callback.register_router_decision_callback",
                side_effect=RuntimeError("boom"),
            ):
                from litellm_llmrouter.startup import register_router_decision_callback

                result = register_router_decision_callback()
                assert result is None


# =============================================================================
# register_strategies
# =============================================================================


class TestRegisterStrategies:
    def test_success(self):
        from litellm_llmrouter.startup import register_strategies

        with patch(
            "litellm_llmrouter.strategies.register_llmrouter_strategies",
            return_value=["knn", "mlp", "svm"],
        ):
            result = register_strategies()
            assert result == ["knn", "mlp", "svm"]

    def test_import_error_returns_empty_list(self):
        from litellm_llmrouter.startup import register_strategies

        with patch(
            "litellm_llmrouter.strategies.register_llmrouter_strategies",
            side_effect=ImportError("no module"),
        ):
            result = register_strategies()
            assert result == []


# =============================================================================
# start_config_sync_if_enabled
# =============================================================================


class TestStartConfigSync:
    def test_disabled_by_default(self):
        from litellm_llmrouter.startup import start_config_sync_if_enabled

        with patch.dict(os.environ, {"CONFIG_HOT_RELOAD": "false"}):
            with patch("litellm_llmrouter.config_sync.start_config_sync") as mock_sync:
                start_config_sync_if_enabled()
                mock_sync.assert_not_called()

    def test_enabled_calls_start(self):
        from litellm_llmrouter.startup import start_config_sync_if_enabled

        with patch.dict(os.environ, {"CONFIG_HOT_RELOAD": "true"}):
            with patch("litellm_llmrouter.config_sync.start_config_sync") as mock_sync:
                start_config_sync_if_enabled()
                mock_sync.assert_called_once()

    def test_import_error_handled(self):
        from litellm_llmrouter.startup import start_config_sync_if_enabled

        with patch.dict(os.environ, {"CONFIG_HOT_RELOAD": "true"}):
            with patch(
                "litellm_llmrouter.config_sync.start_config_sync",
                side_effect=ImportError("no module"),
            ):
                # Should not raise
                start_config_sync_if_enabled()


# =============================================================================
# init_observability_if_enabled
# =============================================================================


class TestInitObservability:
    def test_disabled_via_env(self):
        from litellm_llmrouter.startup import init_observability_if_enabled

        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            with patch(
                "litellm_llmrouter.observability.init_observability"
            ) as mock_init:
                init_observability_if_enabled()
                mock_init.assert_not_called()

    def test_enabled_calls_init(self):
        from litellm_llmrouter.startup import init_observability_if_enabled

        with patch.dict(
            os.environ,
            {
                "OTEL_ENABLED": "true",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
                "OTEL_SERVICE_NAME": "test-service",
            },
        ):
            with patch(
                "litellm_llmrouter.observability.init_observability"
            ) as mock_init:
                init_observability_if_enabled()
                mock_init.assert_called_once_with(
                    service_name="test-service",
                    otlp_endpoint="http://collector:4317",
                    enable_traces=True,
                    enable_logs=True,
                    enable_metrics=True,
                )

    def test_enabled_default_service_name(self):
        from litellm_llmrouter.startup import init_observability_if_enabled

        env = {"OTEL_ENABLED": "true"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("OTEL_SERVICE_NAME", None)
            os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
            with patch(
                "litellm_llmrouter.observability.init_observability"
            ) as mock_init:
                init_observability_if_enabled()
                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs["service_name"] == "litellm-gateway"

    def test_import_error_handled(self):
        from litellm_llmrouter.startup import init_observability_if_enabled

        with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.observability.init_observability",
                side_effect=ImportError("no otel"),
            ):
                # Should not raise
                init_observability_if_enabled()

    def test_generic_exception_handled(self):
        from litellm_llmrouter.startup import init_observability_if_enabled

        with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.observability.init_observability",
                side_effect=RuntimeError("otel failed"),
            ):
                # Should not raise
                init_observability_if_enabled()


# =============================================================================
# init_mcp_tracing_if_enabled
# =============================================================================


class TestInitMCPTracing:
    def test_disabled_by_default(self):
        from litellm_llmrouter.startup import init_mcp_tracing_if_enabled

        with patch.dict(os.environ, {"MCP_GATEWAY_ENABLED": "false"}):
            with patch(
                "litellm_llmrouter.mcp_tracing.instrument_mcp_gateway"
            ) as mock_instr:
                init_mcp_tracing_if_enabled()
                mock_instr.assert_not_called()

    def test_enabled_calls_instrument(self):
        from litellm_llmrouter.startup import init_mcp_tracing_if_enabled

        with patch.dict(os.environ, {"MCP_GATEWAY_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.mcp_tracing.instrument_mcp_gateway",
                return_value=True,
            ) as mock_instr:
                init_mcp_tracing_if_enabled()
                mock_instr.assert_called_once()

    def test_import_error_handled(self):
        from litellm_llmrouter.startup import init_mcp_tracing_if_enabled

        with patch.dict(os.environ, {"MCP_GATEWAY_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.mcp_tracing.instrument_mcp_gateway",
                side_effect=ImportError("no module"),
            ):
                init_mcp_tracing_if_enabled()

    def test_generic_exception_handled(self):
        from litellm_llmrouter.startup import init_mcp_tracing_if_enabled

        with patch.dict(os.environ, {"MCP_GATEWAY_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.mcp_tracing.instrument_mcp_gateway",
                side_effect=RuntimeError("boom"),
            ):
                init_mcp_tracing_if_enabled()


# =============================================================================
# init_a2a_tracing_if_enabled
# =============================================================================


class TestInitA2ATracing:
    def test_always_registers_middleware(self):
        """A2A middleware is always registered regardless of A2A_GATEWAY_ENABLED."""
        from litellm_llmrouter.startup import init_a2a_tracing_if_enabled

        mock_app = MagicMock()
        with patch.dict(os.environ, {"A2A_GATEWAY_ENABLED": "false"}):
            with patch(
                "litellm_llmrouter.a2a_tracing.register_a2a_middleware",
                return_value=True,
            ) as mock_mw:
                init_a2a_tracing_if_enabled(mock_app)
                mock_mw.assert_called_once_with(mock_app)

    def test_gateway_instrumentation_when_enabled(self):
        from litellm_llmrouter.startup import init_a2a_tracing_if_enabled

        mock_app = MagicMock()
        with patch.dict(os.environ, {"A2A_GATEWAY_ENABLED": "true"}):
            with patch(
                "litellm_llmrouter.a2a_tracing.register_a2a_middleware",
                return_value=True,
            ):
                with patch(
                    "litellm_llmrouter.a2a_tracing.instrument_a2a_gateway",
                    return_value=True,
                ) as mock_gw:
                    init_a2a_tracing_if_enabled(mock_app)
                    mock_gw.assert_called_once()

    def test_gateway_not_instrumented_when_disabled(self):
        from litellm_llmrouter.startup import init_a2a_tracing_if_enabled

        mock_app = MagicMock()
        with patch.dict(os.environ, {"A2A_GATEWAY_ENABLED": "false"}):
            with patch(
                "litellm_llmrouter.a2a_tracing.register_a2a_middleware",
                return_value=True,
            ):
                with patch(
                    "litellm_llmrouter.a2a_tracing.instrument_a2a_gateway"
                ) as mock_gw:
                    init_a2a_tracing_if_enabled(mock_app)
                    mock_gw.assert_not_called()

    def test_middleware_import_error_handled(self):
        from litellm_llmrouter.startup import init_a2a_tracing_if_enabled

        mock_app = MagicMock()
        with patch(
            "litellm_llmrouter.a2a_tracing.register_a2a_middleware",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            init_a2a_tracing_if_enabled(mock_app)

    def test_middleware_generic_exception_handled(self):
        from litellm_llmrouter.startup import init_a2a_tracing_if_enabled

        mock_app = MagicMock()
        with patch(
            "litellm_llmrouter.a2a_tracing.register_a2a_middleware",
            side_effect=RuntimeError("boom"),
        ):
            # Should not raise
            init_a2a_tracing_if_enabled(mock_app)


# =============================================================================
# main() - orchestration order and argument parsing
# =============================================================================


class TestMain:
    def _run_main_with_patches(self, argv, **extra_env):
        """Helper to run main() with all external calls mocked."""
        from litellm_llmrouter import startup

        call_order = []

        def track(name):
            def fn(*args, **kwargs):
                call_order.append(name)

            return fn

        env = dict(extra_env)
        # Remove env vars that could interfere
        env.setdefault("LITELLM_PORT", "4000")

        with (
            patch("sys.argv", argv),
            patch.dict(os.environ, env, clear=False),
            patch.object(
                startup,
                "init_observability_if_enabled",
                side_effect=track("observability"),
            ),
            patch.object(
                startup,
                "run_leader_migrations_if_enabled",
                side_effect=track("leader_migrations"),
            ),
            patch.object(
                startup,
                "register_router_decision_callback",
                side_effect=track("router_callback"),
            ),
            patch.object(
                startup,
                "init_mcp_tracing_if_enabled",
                side_effect=track("mcp_tracing"),
            ),
            patch.object(
                startup,
                "register_strategies",
                side_effect=track("strategies"),
            ),
            patch.object(
                startup,
                "start_config_sync_if_enabled",
                side_effect=track("config_sync"),
            ),
            patch.object(
                startup,
                "run_litellm_proxy_inprocess",
                side_effect=track("run_proxy"),
            ) as mock_run,
            patch(
                "litellm_llmrouter.routing_strategy_patch.is_patch_applied",
                return_value=False,
            ),
        ):
            startup.main()
            return call_order, mock_run

    def test_orchestration_order(self):
        """Verify initialization functions are called in the correct order."""
        call_order, _ = self._run_main_with_patches(["startup"])
        assert call_order == [
            "observability",
            "leader_migrations",
            "router_callback",
            "mcp_tracing",
            "strategies",
            "config_sync",
            "run_proxy",
        ]

    def test_default_port(self):
        os.environ.pop("LITELLM_PORT", None)
        os.environ.pop("LITELLM_CONFIG_PATH", None)
        _, mock_run = self._run_main_with_patches(["startup"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["port"] == 4000

    def test_custom_port_via_args(self):
        _, mock_run = self._run_main_with_patches(["startup", "--port", "8080"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["port"] == 8080

    def test_config_path_via_args(self):
        _, mock_run = self._run_main_with_patches(
            ["startup", "--config", "/my/config.yaml"]
        )
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["config_path"] == "/my/config.yaml"

    def test_default_host(self):
        _, mock_run = self._run_main_with_patches(["startup"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"

    def test_custom_host_via_args(self):
        _, mock_run = self._run_main_with_patches(["startup", "--host", "127.0.0.1"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"

    def test_debug_flag(self):
        _, mock_run = self._run_main_with_patches(["startup", "--debug"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["debug"] is True

    def test_workers_default_one(self):
        _, mock_run = self._run_main_with_patches(["startup"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["workers"] == 1

    def test_unknown_args_ignored(self, capsys):
        """Unknown args are noted but don't cause failure."""
        self._run_main_with_patches(["startup", "--unknown-flag", "value"])
        captured = capsys.readouterr()
        assert "unknown" in captured.out.lower()
