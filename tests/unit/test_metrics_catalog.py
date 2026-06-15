"""Contract test for the canonical metric catalog (metrics-1, RouteIQ-2d35).

This test reconciles the "metric-name split-brain": code creates OTel
instruments under dot-style names (``gateway.request.total``) while the Helm
Grafana dashboard queries the Prometheus-exposition names produced AFTER the
OTel Collector's OTel->Prometheus translation (``gateway_request_total``).

``METRIC_CATALOG`` in ``telemetry_contracts.py`` is the single source of truth
mapping each CODE name to its PROMETHEUS name. This test pins three contracts,
mirroring how ``test_routing_decision_log`` asserts ``ROUTING_DECISION_COLUMNS``
is the data-lake column contract:

  (a) every instrument actually created by ``GatewayMetrics`` (plus the
      lazily-created cost-aware histogram in ``strategies.py``) appears in the
      catalog -- introspected from the live instrument ``.name`` attributes, so
      the test FAILS if code adds an instrument absent from the catalog;

  (b) each catalog ``prometheus_name`` equals the independent OTel->Prometheus
      normalization of its ``otel_name`` (dots->underscores, unit suffix,
      ``_total`` only for monotonic counters);

  (c) every PromQL metric name referenced by the shipped Grafana dashboard
      resolves to a catalog ``prometheus_name`` -- so the test FAILS if a panel
      queries a metric the gateway never emits.

WHY (a) and (c) actually catch drift:

  * (a): If someone adds ``self.foo = meter.create_counter("gateway.foo")`` to
    ``GatewayMetrics`` without a catalog entry, ``test_every_created_instrument_is_in_catalog``
    introspects the new instance, sees ``gateway.foo`` is not in
    ``METRIC_CATALOG_OTEL_NAMES``, and fails -- the catalog can never silently
    fall behind the code.

  * (c): If someone adds a dashboard panel querying ``routeiq_made_up_metric``,
    ``test_dashboard_metric_names_resolve_to_catalog`` regexes that token out of
    the YAML, finds it is not a catalog ``prometheus_name`` (after stripping the
    histogram ``_bucket``/``_sum``/``_count`` suffixes), and fails -- a dashboard
    can never reference a name the gateway never exposes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from opentelemetry.metrics import (
    Counter,
    Histogram,
    ObservableGauge,
    UpDownCounter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm_llmrouter.metrics import (
    GatewayMetrics,
    reset_gateway_metrics,
)
from litellm_llmrouter.telemetry_contracts import (
    METRIC_CATALOG,
    METRIC_CATALOG_BY_OTEL_NAME,
    METRIC_CATALOG_OTEL_NAMES,
    METRIC_CATALOG_PROM_NAMES,
    MetricType,
    otel_metric_name_to_prometheus,
)

# Every OTel instrument base type whose instances carry a ``.name`` we catalog.
_OTEL_INSTRUMENT_TYPES = (Counter, UpDownCounter, Histogram, ObservableGauge)

# Histogram series fan out into <name>_bucket / <name>_sum / <name>_count in the
# Prometheus exposition; PromQL references the suffixed series. Strip the suffix
# before resolving a token back to a base catalog name.
_HISTOGRAM_SERIES_SUFFIXES = ("_bucket", "_sum", "_count")

# strategies.py creates this histogram lazily on first cost-aware decision; it
# is not part of GatewayMetrics, so the introspection test allow-lists it as a
# known catalog entry sourced from strategies.py.
_STRATEGIES_LAZY_INSTRUMENTS = frozenset({"routeiq.routing.cost_per_1k_tokens"})


@pytest.fixture(autouse=True)
def _reset_metrics_singleton():
    """Reset the GatewayMetrics singleton around each test (no leak)."""
    reset_gateway_metrics()
    yield
    reset_gateway_metrics()


@pytest.fixture()
def gateway_metrics() -> GatewayMetrics:
    """A GatewayMetrics backed by an in-memory reader (never touches a provider)."""
    provider = MeterProvider(metric_readers=[InMemoryMetricReader()])
    meter = provider.get_meter("test-catalog-meter", "0.1.0")
    return GatewayMetrics(meter)


def _created_instrument_names(metrics: GatewayMetrics) -> set[str]:
    """Introspect every OTel instrument GatewayMetrics actually created.

    Mirrors how ``test_routing_decision_log`` derives the live column set from
    the record rather than trusting a hand-maintained list: we read ``.name``
    off each instrument instance so the set is what the code REALLY built.
    """
    names: set[str] = set()
    for value in vars(metrics).values():
        if isinstance(value, _OTEL_INSTRUMENT_TYPES):
            names.add(value.name)
    return names


# ----------------------------------------------------------- (a) code coverage


def test_every_created_instrument_is_in_catalog(
    gateway_metrics: GatewayMetrics,
) -> None:
    """Every instrument GatewayMetrics creates has a catalog entry.

    FAILS if code adds a ``meter.create_*`` instrument without a MetricSpec.
    """
    created = _created_instrument_names(gateway_metrics)
    # Sanity: introspection found the instruments (not an empty set false-pass).
    assert created, "no OTel instruments introspected from GatewayMetrics"

    missing = sorted(created - METRIC_CATALOG_OTEL_NAMES)
    assert not missing, (
        "GatewayMetrics created instruments absent from METRIC_CATALOG: "
        f"{missing}. Add a MetricSpec in telemetry_contracts.py."
    )


def test_catalog_has_no_orphan_gateway_metrics(
    gateway_metrics: GatewayMetrics,
) -> None:
    """Every catalog entry maps to a real instrument (no dead catalog rows).

    GatewayMetrics-sourced entries must correspond to a live instrument;
    strategies.py-sourced entries are allow-listed (created lazily elsewhere).
    """
    created = _created_instrument_names(gateway_metrics)
    for spec in METRIC_CATALOG:
        if spec.source == "metrics.py":
            assert spec.otel_name in created, (
                f"catalog entry {spec.otel_name!r} (source=metrics.py) has no "
                "matching GatewayMetrics instrument"
            )
        else:
            assert spec.otel_name in _STRATEGIES_LAZY_INSTRUMENTS, (
                f"catalog entry {spec.otel_name!r} has unknown non-metrics.py "
                f"source {spec.source!r}"
            )


def test_catalog_otel_names_are_unique() -> None:
    """No duplicate otel_name / prometheus_name rows in the catalog."""
    otel_names = [s.otel_name for s in METRIC_CATALOG]
    prom_names = [s.prometheus_name for s in METRIC_CATALOG]
    assert len(otel_names) == len(set(otel_names)), "duplicate otel_name in catalog"
    assert len(prom_names) == len(set(prom_names)), "duplicate prometheus_name"
    assert len(METRIC_CATALOG_OTEL_NAMES) == len(METRIC_CATALOG)
    assert len(METRIC_CATALOG_PROM_NAMES) == len(METRIC_CATALOG)


# ------------------------------------------------- (b) normalization round-trip


def test_prometheus_name_is_the_otel_normalization() -> None:
    """Each catalog prometheus_name == OTel->Prometheus normalization of otel_name.

    Independent re-derivation guards against a hand-edited prometheus_name that
    drifts from the documented translation rules.
    """
    for spec in METRIC_CATALOG:
        expected = otel_metric_name_to_prometheus(
            spec.otel_name, spec.metric_type, spec.unit
        )
        assert spec.prometheus_name == expected, (
            f"{spec.otel_name}: catalog prometheus_name {spec.prometheus_name!r} "
            f"!= normalization {expected!r}"
        )


def test_normalization_rules_are_correct() -> None:
    """Pin the three normalization rules on representative cases."""
    # dots -> underscores
    assert (
        otel_metric_name_to_prometheus(
            "gateway.request.active", MetricType.UP_DOWN_COUNTER, "{request}"
        )
        == "gateway_request_active"
    )
    # monotonic counter gets _total; annotation unit dropped
    assert (
        otel_metric_name_to_prometheus(
            "gateway.tokens.total", MetricType.COUNTER, "{token}"
        )
        == "gateway_tokens_total"
    )
    # histogram + seconds unit suffix, no _total
    assert (
        otel_metric_name_to_prometheus(
            "gen_ai.client.operation.duration", MetricType.HISTOGRAM, "s"
        )
        == "gen_ai_client_operation_duration_seconds"
    )
    # unknown unit (USD) appended verbatim+lowercased, plus _total for counter
    assert (
        otel_metric_name_to_prometheus("gateway.cost.total", MetricType.COUNTER, "USD")
        == "gateway_cost_total_usd_total"
    )
    # up_down_counter (non-monotonic) NEVER gets _total
    assert not otel_metric_name_to_prometheus(
        "gateway.request.active", MetricType.UP_DOWN_COUNTER, "{request}"
    ).endswith("_total")


def test_counter_names_end_in_total_histograms_do_not() -> None:
    """The _total suffix is present iff the instrument is a monotonic counter."""
    for spec in METRIC_CATALOG:
        if spec.metric_type is MetricType.COUNTER:
            assert spec.prometheus_name.endswith("_total"), (
                f"counter {spec.otel_name} missing _total suffix"
            )
        else:
            assert not spec.prometheus_name.endswith("_total"), (
                f"non-counter {spec.otel_name} should not carry _total"
            )


# ----------------------------------------------------- (c) dashboard resolution


def _find_dashboard_yaml() -> Path:
    """Locate the shipped Grafana dashboard template within the repo tree."""
    here = Path(__file__).resolve()
    rel = Path("deploy/charts/routeiq-gateway/templates/grafana-dashboard.yaml")
    for parent in here.parents:
        candidate = parent / rel
        if candidate.exists():
            return candidate
    pytest.skip(f"grafana dashboard not found above {here}")


# A PromQL metric name token: starts with a letter/underscore, then word chars
# and colons. We then filter to the RouteIQ metric namespaces so we ignore
# PromQL function names (rate, sum, histogram_quantile, le, ...).
_METRIC_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_:]*")
_ROUTEIQ_NAMESPACES = ("gateway_", "gen_ai_", "quota_", "routeiq_")


def _strip_histogram_suffix(token: str) -> str:
    """Reduce a Prometheus histogram series token to its base metric name."""
    for suffix in _HISTOGRAM_SERIES_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _dashboard_metric_tokens(text: str) -> set[str]:
    """Extract candidate RouteIQ metric-name tokens from the dashboard JSON."""
    tokens: set[str] = set()
    for match in _METRIC_TOKEN.finditer(text):
        token = match.group(0)
        if token.startswith(_ROUTEIQ_NAMESPACES):
            tokens.add(token)
    return tokens


def test_dashboard_metric_names_resolve_to_catalog() -> None:
    """Every RouteIQ metric the Grafana dashboard queries is in the catalog.

    FAILS if a panel references a Prometheus name the gateway never exposes
    (e.g. a stale ``routeiq_*`` name or a typo'd ``gateway_*`` name).
    """
    dashboard = _find_dashboard_yaml()
    tokens = _dashboard_metric_tokens(dashboard.read_text())
    assert tokens, "no RouteIQ metric tokens found in dashboard (parse failure?)"

    unresolved: set[str] = set()
    for token in tokens:
        base = _strip_histogram_suffix(token)
        if base not in METRIC_CATALOG_PROM_NAMES:
            unresolved.add(token)

    assert not unresolved, (
        "Grafana dashboard references metric names not in METRIC_CATALOG: "
        f"{sorted(unresolved)}. Either fix the PromQL to a catalog "
        "prometheus_name or add the instrument + catalog entry."
    )


def test_dashboard_uses_no_legacy_routeiq_prefixed_metrics() -> None:
    """No panel still uses the invented ``routeiq_*`` exposition names.

    The gateway never emits a ``routeiq_*`` Prometheus metric (the only
    ``routeiq.*`` instrument normalizes to ``routeiq_routing_cost_per_1k_tokens_usd``,
    not queried by this dashboard). Any surviving ``routeiq_`` token is the old
    split-brain name.
    """
    dashboard = _find_dashboard_yaml()
    tokens = _dashboard_metric_tokens(dashboard.read_text())
    legacy = sorted(
        t
        for t in tokens
        if t.startswith("routeiq_")
        and _strip_histogram_suffix(t) not in METRIC_CATALOG_PROM_NAMES
    )
    assert not legacy, f"dashboard still references legacy routeiq_* metrics: {legacy}"


def test_catalog_lookup_indexes_are_consistent() -> None:
    """The by-name lookup dicts agree with METRIC_CATALOG."""
    for spec in METRIC_CATALOG:
        assert METRIC_CATALOG_BY_OTEL_NAME[spec.otel_name] is spec
    assert set(METRIC_CATALOG_BY_OTEL_NAME) == set(METRIC_CATALOG_OTEL_NAMES)
