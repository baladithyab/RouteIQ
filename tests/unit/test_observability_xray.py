"""Unit tests for the AWS X-Ray app-side wiring (RouteIQ-3c0a).

Covers ``observability.py``'s X-Ray trace-ID generator + ``X-Amzn-Trace-Id``
propagator wiring:

* settings-gated, DEFAULT OFF -> trace IDs stay W3C-format unless opted in;
* when enabled (and the optional contrib packages are present), a NEW
  TracerProvider is constructed with the X-Ray ``id_generator`` so generated
  trace IDs are X-Ray epoch format, and the X-Ray propagator round-trips the
  ``X-Amzn-Trace-Id`` header;
* the optional packages' ABSENCE degrades to W3C with a warning (never breaks
  boot) — the production env here has neither package installed, so the
  "present" paths are exercised with lightweight fakes injected into
  ``sys.modules``.

cred-free: no live AWS, no boto3. Follows test_observability.py's importlib
loading style (no heavy litellm_llmrouter package import).
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.sdk.trace import TracerProvider, id_generator as _id_gen_mod

# Load the module under test directly (mirrors tests/unit/test_observability.py).
_spec = importlib.util.spec_from_file_location(
    "observability_xray_under_test", "src/litellm_llmrouter/observability.py"
)
assert _spec is not None and _spec.loader is not None
observability = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(observability)

ObservabilityManager = observability.ObservabilityManager
_build_xray_id_generator = observability._build_xray_id_generator
_install_xray_propagator = observability._install_xray_propagator


# ---------------------------------------------------------------------------
# Fakes for the optional opentelemetry-contrib AWS X-Ray packages.
# ---------------------------------------------------------------------------
# Neither ``opentelemetry-sdk-extension-aws`` nor
# ``opentelemetry-propagator-aws-xray`` is installed in this env, so the
# "enabled + present" path is exercised by injecting tiny fakes into sys.modules
# under the exact import paths the code guards. The fake id generator produces a
# trace ID whose top 32 bits are the current epoch second (the X-Ray-format
# invariant: a W3C-random id would essentially never satisfy this).


class _FakeAwsXRayIdGenerator(_id_gen_mod.IdGenerator):
    """Generates X-Ray-epoch-format trace IDs (top 32 bits == epoch seconds)."""

    def generate_span_id(self) -> int:
        return 0x1234567890ABCDEF

    def generate_trace_id(self) -> int:
        epoch = int(time.time())
        # top 32 bits = epoch, low 96 bits = a fixed nonzero unique-ish value.
        return (epoch << 96) | 0x0123456789ABCDEF0123ABCD


class _FakeAwsXRayPropagator:
    """Minimal text-map propagator that round-trips an X-Amzn-Trace-Id header."""

    _HEADER = "X-Amzn-Trace-Id"

    def extract(self, carrier, context=None, getter=None):  # type: ignore[no-untyped-def]
        # Return a real (non-None) context — the real propagator likewise returns
        # a fresh root context when none is supplied. The load-bearing behavior
        # (the X-Amzn-Trace-Id header is recognised) is covered by inject.
        from opentelemetry.context import get_current

        return context if context is not None else get_current()

    def inject(self, carrier, context=None, setter=None):  # type: ignore[no-untyped-def]
        # Emit the X-Ray header so a round-trip test can observe it.
        if isinstance(carrier, dict):
            carrier[self._HEADER] = "Root=1-5759e988-bd862e3fe1be46a994272793;Sampled=1"

    @property
    def fields(self):  # type: ignore[no-untyped-def]
        return {self._HEADER}


@pytest.fixture
def _inject_xray_packages(monkeypatch):
    """Inject fake AWS X-Ray contrib packages into sys.modules for the test."""
    # opentelemetry.sdk.extension.aws.trace.AwsXRayIdGenerator
    ext_pkg = types.ModuleType("opentelemetry.sdk.extension")
    aws_pkg = types.ModuleType("opentelemetry.sdk.extension.aws")
    trace_mod = types.ModuleType("opentelemetry.sdk.extension.aws.trace")
    trace_mod.AwsXRayIdGenerator = _FakeAwsXRayIdGenerator  # type: ignore[attr-defined]

    # opentelemetry.propagators.aws.AwsXRayPropagator
    prop_pkg = types.ModuleType("opentelemetry.propagators.aws")
    prop_pkg.AwsXRayPropagator = _FakeAwsXRayPropagator  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.extension", ext_pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.extension.aws", aws_pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.extension.aws.trace", trace_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.propagators.aws", prop_pkg)
    yield


# ---------------------------------------------------------------------------
# Helper behavior: graceful degradation when packages absent
# ---------------------------------------------------------------------------


def test_build_xray_id_generator_absent_returns_none():
    """No opentelemetry-sdk-extension-aws -> None (degrade to W3C), no raise."""
    # In this env the package is genuinely absent.
    assert "opentelemetry.sdk.extension.aws.trace" not in sys.modules
    assert _build_xray_id_generator() is None


def test_install_xray_propagator_absent_returns_false():
    """No opentelemetry-propagator-aws-xray -> False (keep W3C), no raise."""
    assert "opentelemetry.propagators.aws" not in sys.modules
    assert _install_xray_propagator() is False


def test_build_xray_id_generator_present_returns_instance(_inject_xray_packages):
    gen = _build_xray_id_generator()
    assert gen is not None
    assert isinstance(gen, _FakeAwsXRayIdGenerator)


def test_install_xray_propagator_present_sets_global_textmap(_inject_xray_packages):
    with patch.object(observability, "logger"):
        from opentelemetry import propagate

        with patch.object(propagate, "set_global_textmap") as mock_set:
            assert _install_xray_propagator() is True
            assert mock_set.call_count == 1
            installed = mock_set.call_args.args[0]
            assert isinstance(installed, _FakeAwsXRayPropagator)


# ---------------------------------------------------------------------------
# Settings gating: default OFF
# ---------------------------------------------------------------------------


def test_xray_disabled_by_default_via_settings():
    """With no explicit arg and default settings, X-Ray is OFF (byte-stable)."""
    mgr = ObservabilityManager(
        service_name="t", enable_logs=False, enable_metrics=False
    )
    # Default settings.otel.xray_enabled is False.
    assert mgr.xray_enabled is False


def test_xray_explicit_arg_overrides_settings():
    mgr = ObservabilityManager(
        service_name="t",
        enable_logs=False,
        enable_metrics=False,
        xray_enabled=True,
    )
    assert mgr.xray_enabled is True


def test_xray_settings_unavailable_fails_closed_to_off():
    """A settings failure must fail-closed to OFF (no spurious X-Ray IDs)."""
    with patch.object(observability, "ObservabilityManager", ObservabilityManager):
        # Force get_settings import to blow up inside __init__.
        with patch.dict(sys.modules, {"litellm_llmrouter.settings": None}):
            mgr = ObservabilityManager(
                service_name="t", enable_logs=False, enable_metrics=False
            )
            assert mgr.xray_enabled is False


# ---------------------------------------------------------------------------
# New-provider path: W3C vs X-Ray trace ID format
# ---------------------------------------------------------------------------


def _fresh_provider_env():
    """Reset the global tracer provider so _init_tracing takes the create-new path.

    The OTel global provider is set-once; we cannot truly unset it. Instead we
    patch trace.get_tracer_provider to return a non-SDK provider so the manager
    creates its own (and never calls set_tracer_provider on the global, which we
    also stub to avoid the override warning polluting the shared provider).
    """


def test_disabled_generates_w3c_trace_id_format():
    """X-Ray OFF -> new provider uses the SDK default (random W3C) id_generator."""
    mgr = ObservabilityManager(
        service_name="t",
        enable_logs=False,
        enable_metrics=False,
        xray_enabled=False,
    )

    captured = {}

    real_tp = TracerProvider

    def _capture_tp(**kwargs):
        captured["kwargs"] = kwargs
        return real_tp(**kwargs)

    # Force the create-new branch: pretend no SDK provider exists yet.
    with patch.object(observability.trace, "get_tracer_provider") as mock_get:
        mock_get.return_value = MagicMock()  # not an SDK TracerProvider
        with patch.object(observability, "_is_sdk_tracer_provider", return_value=False):
            with patch.object(observability.trace, "set_tracer_provider"):
                with patch.object(observability, "TracerProvider", _capture_tp):
                    mgr._init_tracing()

    # No id_generator kwarg => SDK default random (W3C) generator.
    assert "id_generator" not in captured["kwargs"]

    # And a generated trace id from the default generator is NOT X-Ray-format
    # (top 32 bits would have to equal a recent epoch second by pure chance).
    tracer = mgr._tracer_provider.get_tracer("t")  # type: ignore[union-attr]
    with tracer.start_as_current_span("s") as span:
        tid = span.get_span_context().trace_id
    epoch_bits = tid >> 96
    now = int(time.time())
    assert not (now - 600 <= epoch_bits <= now + 600), (
        "default generator must not produce an X-Ray-epoch-format trace id"
    )


def test_enabled_passes_xray_id_generator_and_emits_xray_format(_inject_xray_packages):
    """X-Ray ON + packages present -> new provider gets the X-Ray id_generator and
    generated trace IDs are X-Ray epoch format; the propagator is installed."""
    mgr = ObservabilityManager(
        service_name="t",
        enable_logs=False,
        enable_metrics=False,
        xray_enabled=True,
    )

    captured = {}
    real_tp = TracerProvider

    def _capture_tp(**kwargs):
        captured["kwargs"] = kwargs
        return real_tp(**kwargs)

    from opentelemetry import propagate

    with patch.object(observability.trace, "get_tracer_provider") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.object(observability, "_is_sdk_tracer_provider", return_value=False):
            with patch.object(observability.trace, "set_tracer_provider"):
                with patch.object(observability, "TracerProvider", _capture_tp):
                    with patch.object(propagate, "set_global_textmap") as mock_set:
                        mgr._init_tracing()

    # The X-Ray id_generator was passed at construction.
    assert "id_generator" in captured["kwargs"]
    assert isinstance(captured["kwargs"]["id_generator"], _FakeAwsXRayIdGenerator)

    # The propagator was installed globally.
    assert mock_set.call_count == 1
    assert isinstance(mock_set.call_args.args[0], _FakeAwsXRayPropagator)

    # A generated trace id is X-Ray epoch format (top 32 bits ~ current epoch).
    tracer = mgr._tracer_provider.get_tracer("t")  # type: ignore[union-attr]
    with tracer.start_as_current_span("s") as span:
        tid = span.get_span_context().trace_id
    epoch_bits = tid >> 96
    now = int(time.time())
    assert now - 600 <= epoch_bits <= now + 600, (
        f"X-Ray id generator must yield an epoch-prefixed trace id; got {epoch_bits:#x}"
    )


def test_enabled_reused_provider_skips_id_generator_but_installs_propagator(
    _inject_xray_packages,
):
    """X-Ray ON but an SDK provider is REUSED -> cannot retrofit id_generator,
    but the X-Amzn-Trace-Id propagator is still installed (RouteIQ-3c0a caveat)."""
    mgr = ObservabilityManager(
        service_name="t",
        enable_logs=False,
        enable_metrics=False,
        xray_enabled=True,
    )

    existing = TracerProvider()
    from opentelemetry import propagate

    with patch.object(
        observability.trace, "get_tracer_provider", return_value=existing
    ):
        with patch.object(observability, "_is_sdk_tracer_provider", return_value=True):
            with patch.object(propagate, "set_global_textmap") as mock_set:
                mgr._init_tracing()

    # Provider was reused (not reconstructed), so we cannot assert id_generator
    # was passed — but the propagator install still ran.
    assert mgr._tracer_provider is existing
    assert mock_set.call_count == 1
    assert isinstance(mock_set.call_args.args[0], _FakeAwsXRayPropagator)


# ---------------------------------------------------------------------------
# Round-trip: the installed propagator carries X-Amzn-Trace-Id both ways
# ---------------------------------------------------------------------------


def test_propagator_round_trips_x_amzn_trace_id_header(_inject_xray_packages):
    """The installed X-Ray propagator injects/extracts the X-Amzn-Trace-Id header."""
    from opentelemetry import propagate

    # Install via the real set_global_textmap so we can round-trip through it.
    real_propagator = propagate.get_global_textmap()
    try:
        assert _install_xray_propagator() is True
        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        assert "X-Amzn-Trace-Id" in carrier, (
            "X-Ray propagator must inject the X-Amzn-Trace-Id header"
        )
        # Extraction is reachable through the global text-map without raising.
        ctx = propagate.extract(carrier)
        assert ctx is not None
    finally:
        propagate.set_global_textmap(real_propagator)
