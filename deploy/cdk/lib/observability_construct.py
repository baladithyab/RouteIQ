"""ObservabilityConstruct for the RouteIQ P2 observability stack.

Owns the routing-telemetry observability surface that lands at P2 (ADR-0027):

  - an Amazon Managed Prometheus (AMP) ``aps.CfnWorkspace`` (the scrape sink for
    a future ADOT remote-write path; the pod-role ``aps:RemoteWrite`` grant is
    LATE-BOUND by the composition root, see ``amp_remote_write_grant`` below);
  - a FLAG-GATED-OFF Amazon Managed Grafana (AMG) ``grafana.CfnWorkspace`` +
    its data-source IAM role (default off: AMG requires IAM Identity Center
    (AWS_SSO) already enabled in the account or the workspace fails at deploy);
  - the single TLS-enforced SNS on-call topic (DenyInsecureTransport resource
    policy, SNS-managed key);
  - the CloudWatch Logs metric filters over the IMPORTED P0 routing log group:
    an aggregate latency filter, the PER-MODEL dimensioned latency filter (the
    load-bearing P2 deliverable, keyed on the OTel ``gen_ai.response.model``
    field), and the aggregate router-error-count filter;
  - the CW-Logs-native alarms that read those filters (routing-latency ceiling,
    router-error-count, routing-latency anomaly detection), each wired to BOTH
    an alarm action and an OK action on the on-call topic;
  - a modest secondary CloudWatch dashboard (per-model latency + request-volume
    via a SEARCH() expansion). AMG remains the primary SRE surface when enabled.

WHAT THIS PORT TRIMS FROM THE VSR ``ObservabilityConstruct``:
This is a port of vllm-sr-on-aws/cdk/lib/observability_construct.py, re-derived
symbol-by-symbol from the real source, with the RouteIQ divergences below. The
VSR ECS-only surface is DROPPED: the cdk-monitoring-constructs ``MonitoringFacade``
(ALB/Fargate alarms), the EMF auxiliary log group, the ADOT s3-asset, the
blue/green ``llm_*`` rollback alarms (p99/jailbreak/hallucination/semantic-cache),
the opus-share dashboard + scheduled-Lambda gauge, and the saved Logs-Insights
QueryDefinitions. Those depend on ECS + a running router + cdk-monitoring-constructs
(which P0 deliberately omitted from requirements). What is KEPT is the routing
metric filters + the CW-Logs-native alarms that read them.

THE CRITICAL RouteIQ DIVERGENCE (the per-model dimension key):
VSR's router emits a flat ``$.selected_model`` field; RouteIQ is OpenTelemetry
``gen_ai.*``-semantic-convention native and does NOT emit ``selected_model``.
The telemetry contract emits ``gen_ai.response.model``
(``telemetry_contracts.py:673``). So the per-model dimensioned MetricFilter MUST
be re-keyed to the OTel field, bracket-quoted because the key contains dots:

    dimensions={"model": '$.["gen_ai.response.model"]'}   # NOT $.selected_model

AWS forbids ``default_value`` on a dimensioned filter, so the per-model filter
omits it; the aggregate (no-dimension) filters keep ``default_value`` and remain
the series the alarms read.

DUAL-KEY RECONCILIATION (build-outline D3 + D5):
The structured ``routing_decision`` JSON log line that feeds these filters (a P2
BUILD-NEW app-side emitter in ``router_decision_callback.py``) emits BOTH a
top-level ``selected_model`` key (the data-lake's identity-SerDe column) AND a
top-level ``gen_ai.response.model`` key (the CW dimension key this filter reads),
with the SAME value. The filter below keys on ``gen_ai.response.model`` (the task
mandate); the lake's ``selected_model`` column reads the sibling key. The two
contracts hold simultaneously off one line.

THE CDK-CREATED-LOG-GROUP-BEFORE-METRICFILTER ORDERING (VSR lesson #8):
A CFN ``AWS::Logs::MetricFilter`` requires its target log group to ALREADY EXIST
at deploy time. P0 already CDK-owns the routing log group
(``/aws/containerinsights/<cluster>/routeiq-routing``, ``EksClusterConstruct``),
so this construct does NOT create a second group and does NOT import by
``from_lookup`` (which needs creds). The composition root re-imports the P0 group
by NAME via ``logs.LogGroup.from_log_group_name(...)`` and passes the
``ILogGroup`` in as the ``routing_log_group`` prop. The MetricFilters + the
data-lake SubscriptionFilter attach to that imported group. (The Fluent Bit
pipeline that promotes the router JSON to the CW record top level is a chart /
manifest concern, tracked as a follow-up seed.)

ASCII / Latin-1-only IAM Descriptions (P0 section 4.5): an em-dash (U+2014) passes
``cdk synth`` but FAILS the IAM CREATE API. Every Description here stays ASCII.

Public attributes (the frozen contract the composition root + outputs consume):

    amp_workspace: aps.CfnWorkspace
    amp_workspace_arn: str
    amp_remote_write_url: str
    amg_workspace: grafana.CfnWorkspace | None
    amg_role: iam.Role | None
    alarm_topic: sns.Topic
    routing_latency_filter: logs.MetricFilter
    routing_latency_by_model_filter: logs.MetricFilter
    router_error_filter: logs.MetricFilter
    routing_latency_alarm: cloudwatch.Alarm
    router_error_alarm: cloudwatch.Alarm
    routing_latency_anomaly_alarm: cloudwatch.AnomalyDetectionAlarm
    routing_dashboard: cloudwatch.Dashboard
"""

from __future__ import annotations

from aws_cdk import Aws, Duration
from aws_cdk import aws_aps as aps
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_grafana as grafana
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as sns_subscriptions
from constructs import Construct

# The structured routing-decision marker. RouteIQ's flat JSON line carries a
# top-level ``event`` key with the literal value ``routing_decision`` (the
# emitter's ``_COLUMNS`` contract, build-outline D5) - NOT VSR's ``$.msg``.
_ROUTING_EVENT_FIELD = "$.event"
_ROUTING_EVENT_VALUE = "routing_decision"

# The latency value the structured line carries. RouteIQ emits a flat
# ``latency_ms`` key (int milliseconds) - NOT VSR's ``$.routing_latency_ms``.
_ROUTING_LATENCY_FIELD = "$.latency_ms"

# The MetricFilter JSON dimension key for the per-model latency filter.
# CRITICAL (ADR-0027 / telemetry_contracts.py:673): RouteIQ does NOT emit a flat
# ``selected_model`` field - the telemetry contract emits ``gen_ai.response.model``.
# A filter keyed on ``$.selected_model`` matches ZERO RouteIQ events. The OTel key
# contains dots, so it MUST be bracket-quoted: ``$.["gen_ai.response.model"]`` (the
# plain ``$.gen_ai.response.model`` form parses as nested fields and matches
# nothing). The VSR source uses ``$.selected_model``; the RouteIQ port uses the key
# below. AWS forbids ``default_value`` on a dimensioned filter, so it is omitted.
_ROUTING_MODEL_DIM_KEY = '$.["gen_ai.response.model"]'

# Severity-level selector for the aggregate router-error-count filter. The
# gateway's error path emits a dedicated structured JSON line (RouteIQ-731c,
# observability.py ``emit_error_log``) on the SAME routing log group carrying a
# TOP-LEVEL LOWERCASED ``level == "error"`` literal. NOTE: this is NOT Python's
# default ``levelname`` (which is UPPERCASE ``ERROR``) - the emitter writes the
# lowercased literal precisely so this filter pattern matches. Before that
# emitter shipped, no line carried a ``level`` key and this filter (and its
# alarm) was inert.
_ERROR_LEVEL_FIELD = "$.level"
_ERROR_LEVEL_VALUE = "error"


class ObservabilityConstruct(Construct):
    """AMP + flag-gated AMG + TLS SNS + routing metric filters/alarms (P2)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        routing_log_group: logs.ILogGroup,
        enable_amg: bool = False,
        notify_emails: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self.enable_amg = enable_amg
        # The owned-namespace for the routing CW metrics. Double-quoted in any
        # SEARCH() schema (it contains ``/`` + ``-``); kept as a bare string here
        # for MetricFilter.metric_namespace and Metric.namespace.
        self._routing_ns = f"routeiq/{env_name}/router"

        # -- 1. AMP workspace -------------------------------------------------
        # Created with ONLY an alias; CFN does not return the remote-write URL, so
        # it is composed from the workspace-id token (resolves at deploy time).
        self.amp_workspace = aps.CfnWorkspace(
            self,
            "AmpWorkspace",
            alias=f"routeiq-{env_name}",
        )
        self.amp_workspace_arn = self.amp_workspace.attr_arn
        self.amp_remote_write_url = (
            f"https://aps-workspaces.{Aws.REGION}.amazonaws.com"
            f"/workspaces/{self.amp_workspace.attr_workspace_id}/api/v1/remote_write"
        )

        # -- 2. AMG workspace (FLAG-GATED OFF by default) ---------------------
        # AMG requires IAM Identity Center (AWS_SSO) already enabled in the target
        # account, otherwise CfnWorkspace creation fails at deploy time (README
        # documents this prerequisite). Default flag-off so a deploy without SSO
        # succeeds; when off both amg_role and amg_workspace are None and the
        # nag-suppression helper getattr-guards them. Even CURRENT_ACCOUNT +
        # SERVICE_MANAGED still needs the explicit aps/cloudwatch/xray read grants
        # via role_arn, hence the data-source role.
        self.amg_role: iam.Role | None
        self.amg_workspace: grafana.CfnWorkspace | None
        if enable_amg:
            self.amg_role = iam.Role(
                self,
                "AmgWorkspaceRole",
                assumed_by=iam.ServicePrincipal("grafana.amazonaws.com"),
                description=(
                    f"RouteIQ {env_name} Amazon Managed Grafana data-source "
                    "read role (AMP query + CloudWatch + X-Ray)"
                ),
            )
            self.amg_role.add_to_policy(
                iam.PolicyStatement(
                    sid="AmpQuery",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "aps:QueryMetrics",
                        "aps:GetLabels",
                        "aps:GetSeries",
                        "aps:GetMetricMetadata",
                    ],
                    resources=[self.amp_workspace_arn],
                )
            )
            self.amg_role.add_to_policy(
                iam.PolicyStatement(
                    sid="CloudWatchRead",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "cloudwatch:GetMetricData",
                        "cloudwatch:ListMetrics",
                        "cloudwatch:GetMetricStatistics",
                        "cloudwatch:DescribeAlarms",
                    ],
                    resources=["*"],
                )
            )
            self.amg_role.add_to_policy(
                iam.PolicyStatement(
                    sid="XRayRead",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "xray:GetTraceSummaries",
                        "xray:BatchGetTraces",
                        "xray:GetServiceGraph",
                    ],
                    resources=["*"],
                )
            )
            self.amg_workspace = grafana.CfnWorkspace(
                self,
                "AmgWorkspace",
                account_access_type="CURRENT_ACCOUNT",
                authentication_providers=["AWS_SSO"],
                permission_type="SERVICE_MANAGED",
                data_sources=["PROMETHEUS", "XRAY", "CLOUDWATCH"],
                notification_destinations=["SNS"],
                name=f"routeiq-{env_name}",
                role_arn=self.amg_role.role_arn,
            )
        else:
            self.amg_role = None
            self.amg_workspace = None

        # -- 3. SNS on-call topic (TLS-enforced) ------------------------------
        # The single oncall topic every alarm pages. SNS-managed key (alias/aws/sns),
        # NOT a CMK: CloudWatch Alarms CANNOT publish to a CMK-encrypted topic
        # without a key policy granting cloudwatch.amazonaws.com, and these alarm
        # payloads carry no sensitive data (metric name + state), so the managed key
        # is correct and avoids the CMK-publish-denied footgun. enforce_ssl: a
        # DenyInsecureTransport resource policy denies non-TLS publish/subscribe.
        self.alarm_topic = sns.Topic(
            self,
            "AlarmTopic",
            topic_name=f"routeiq-{env_name}-oncall",
            display_name=f"routeiq {env_name} oncall",
        )
        self.alarm_topic.add_to_resource_policy(
            iam.PolicyStatement(
                sid="DenyInsecureTransport",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["sns:Publish", "sns:Subscribe"],
                resources=[self.alarm_topic.topic_arn],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
            )
        )
        # Optional auto-subscribe of operator emails. Default: NO subscription
        # (operator runs ``aws sns subscribe`` post-deploy). Each subscription is
        # a confirmation-pending email until the recipient confirms.
        for email in notify_emails or []:
            self.alarm_topic.add_subscription(sns_subscriptions.EmailSubscription(email))

        # -- 4. CW Logs metric filters (over the IMPORTED P0 routing group) ---
        # (a) Aggregate routing latency (NO dimensions -> keeps default_value=0).
        # This is the series the latency + anomaly alarms read. The structured
        # line carries top-level ``$.event == "routing_decision"`` + ``$.latency_ms``
        # (matching the emitter _COLUMNS, NOT VSR's $.msg / $.routing_latency_ms).
        self.routing_latency_filter = logs.MetricFilter(
            self,
            "RoutingLatencyFilter",
            log_group=routing_log_group,
            filter_pattern=logs.FilterPattern.string_value(
                _ROUTING_EVENT_FIELD, "=", _ROUTING_EVENT_VALUE
            ),
            metric_namespace=self._routing_ns,
            metric_name="routing_latency_ms",
            metric_value=_ROUTING_LATENCY_FIELD,
            unit=cloudwatch.Unit.MILLISECONDS,
            default_value=0,
        )

        # (b) PER-MODEL DIMENSIONED routing latency (the load-bearing P2 filter).
        # A CW Logs metric filter supports up to 3 dimensions keyed on JSON fields,
        # so a second filter dimensioned by the response model emits a SEPARATE
        # routing_latency_ms_by_model metric per model - no Prometheus/AMP needed.
        # Cardinality is bounded by RouteIQ's CLOSED model catalog (a few dozen
        # arms, not unbounded like request_id), so this is safe. The dimension key
        # MUST be the OTel gen_ai.response.model field (NOT selected_model, which
        # RouteIQ never emits), bracket-quoted because it contains dots. AWS forbids
        # default_value on a dimensioned filter, so this one OMITS it (the aggregate
        # filter above keeps the no-dimension form for the alarms).
        self.routing_latency_by_model_filter = logs.MetricFilter(
            self,
            "RoutingLatencyByModelFilter",
            log_group=routing_log_group,
            filter_pattern=logs.FilterPattern.string_value(
                _ROUTING_EVENT_FIELD, "=", _ROUTING_EVENT_VALUE
            ),
            metric_namespace=self._routing_ns,
            metric_name="routing_latency_ms_by_model",
            metric_value=_ROUTING_LATENCY_FIELD,
            unit=cloudwatch.Unit.MILLISECONDS,
            dimensions={"model": _ROUTING_MODEL_DIM_KEY},
        )

        # (c) Aggregate router error count (NO dimensions -> keeps default_value=0).
        # Reads the gateway's general error log lines in the same group (top-level
        # ``$.level == "error"``). A continuous 0 baseline makes "went non-zero"
        # crisp for the alarm.
        self.router_error_filter = logs.MetricFilter(
            self,
            "RouterErrorFilter",
            log_group=routing_log_group,
            filter_pattern=logs.FilterPattern.string_value(
                _ERROR_LEVEL_FIELD, "=", _ERROR_LEVEL_VALUE
            ),
            metric_namespace=self._routing_ns,
            metric_name="router_error_log_count",
            metric_value="1",
            default_value=0,
        )

        # -- 5. CW-Logs-native alarms (read the aggregate filters above) ------
        # These do NOT need a running router/AMP - they read CW Logs metrics that
        # exist as soon as traffic flows. Each alarm gets BOTH an alarm action AND
        # an OK action so recovery clears (the VSR regression-fix lesson: the EKS
        # alarms originally had no action and would page no one).
        self._build_alarms(env_name)

        # -- 6. Secondary CloudWatch dashboard --------------------------------
        # Per-model latency + request-volume via a SEARCH() expansion. Because the
        # model dimension VALUES are runtime-learned (not known at synth), a static
        # Metric(dimensions_map=...) cannot list them - the SEARCH() expands to one
        # series per discovered model at render time. Request VOLUME is derived at
        # ZERO new resources: every routing_decision line emits exactly one
        # routing_latency_ms_by_model sample, so the SampleCount statistic of the
        # latency metric IS the request count. AMG is primary when enabled; this
        # dashboard is the secondary CloudWatch view.
        self.routing_dashboard = self._build_dashboard(env_name)

    def _build_alarms(self, env_name: str) -> None:
        """Build the CW-Logs-native routing alarms and wire SNS actions.

        Sets ``self.routing_latency_alarm``, ``self.router_error_alarm``, and
        ``self.routing_latency_anomaly_alarm``. Every alarm pages the single
        on-call topic and clears on recovery (alarm + OK actions).
        """
        # (1) Routing latency ceiling. Maximum of the aggregate latency metric;
        # a 30s absolute ceiling (no baseline) catches a wedged routing path /
        # large-prompt timeout / backend stall. NOT_BREACHING: missing data is
        # not a latency signal.
        latency_metric = cloudwatch.Metric(
            namespace=self._routing_ns,
            metric_name="routing_latency_ms",
            statistic="Maximum",
            period=Duration.minutes(1),
        )
        self.routing_latency_alarm = cloudwatch.Alarm(
            self,
            "RoutingLatencyAlarm",
            alarm_name=f"routeiq-{env_name}-routing-latency-high",
            alarm_description=(
                f"RouteIQ {env_name} routing latency exceeded 30000 ms (30s "
                "absolute ceiling) for 3 consecutive minutes - a wedged routing "
                "path, a large-prompt timeout, or a backend stall. Reads the "
                "aggregate routing_latency_ms CW Logs metric filter."
            ),
            metric=latency_metric,
            threshold=30000,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=3,
            datapoints_to_alarm=3,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # (2) Router error-log count. Sum over 5min of the error-line counter;
        # threshold 10 errors over the window. NOT_BREACHING.
        error_metric = cloudwatch.Metric(
            namespace=self._routing_ns,
            metric_name="router_error_log_count",
            statistic="Sum",
            period=Duration.minutes(5),
        )
        self.router_error_alarm = cloudwatch.Alarm(
            self,
            "RouterErrorAlarm",
            alarm_name=f"routeiq-{env_name}-router-error-log-count",
            alarm_description=(
                f"RouteIQ {env_name} router error-log count exceeded 10 events "
                "over 5 minutes (top-level level=error lines on the routing log "
                "group). Reads the router_error_log_count CW Logs metric filter."
            ),
            metric=error_metric,
            threshold=10,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            datapoints_to_alarm=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # (3) Routing-latency anomaly detection. Self-training band on the average
        # routing latency; std_devs=3 (a wider band -> fewer false pages on a
        # noisy single-replica signal), only abnormally HIGH latency pages
        # (GREATER_THAN_UPPER_THRESHOLD). Reads INSUFFICIENT_DATA until ~2wk of
        # traffic trains the band, then alarms - the baseline-free "p99 abnormal"
        # signal the declined AMP scrape path would otherwise have given.
        anomaly_metric = cloudwatch.Metric(
            namespace=self._routing_ns,
            metric_name="routing_latency_ms",
            statistic="Average",
            period=Duration.minutes(5),
        )
        self.routing_latency_anomaly_alarm = cloudwatch.AnomalyDetectionAlarm(
            self,
            "RoutingLatencyAnomalyAlarm",
            alarm_name=f"routeiq-{env_name}-routing-latency-anomaly",
            alarm_description=(
                f"RouteIQ {env_name} routing latency is abnormally HIGH vs its "
                "self-trained 3-sigma band (anomaly detection on average "
                "routing_latency_ms over 5min). Reads INSUFFICIENT_DATA until the "
                "band trains on ~2 weeks of traffic."
            ),
            metric=anomaly_metric,
            std_devs=3,
            evaluation_periods=3,
            comparison_operator=(cloudwatch.ComparisonOperator.GREATER_THAN_UPPER_THRESHOLD),
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # Wire BOTH an alarm action and an OK action on every alarm so recovery
        # clears. Loop over the set (the VSR regression-fix lesson: an alarm with
        # no action sits in ALARM state and pages no one).
        action = cw_actions.SnsAction(self.alarm_topic)
        for alarm in (
            self.routing_latency_alarm,
            self.router_error_alarm,
            self.routing_latency_anomaly_alarm,
        ):
            alarm.add_alarm_action(action)
            alarm.add_ok_action(action)

    def _build_dashboard(self, env_name: str) -> cloudwatch.Dashboard:
        """Build the secondary per-model routing dashboard (SEARCH() expansion).

        Two math widgets reading routing_latency_ms_by_model: per-model latency
        (Average) and per-model request volume (SampleCount). The namespace is
        DOUBLE-QUOTED inside the SEARCH schema because it contains ``/`` and ``-``;
        the bare form would mis-parse.
        """
        per_model_latency = cloudwatch.MathExpression(
            expression=(
                f'SEARCH(\'{{"{self._routing_ns}",model}} '
                "MetricName=\"routing_latency_ms_by_model\"', 'Average', 60)"
            ),
            label="",  # SEARCH supplies per-series labels (the model name)
            period=Duration.minutes(1),
        )
        per_model_volume = cloudwatch.MathExpression(
            expression=(
                f'SEARCH(\'{{"{self._routing_ns}",model}} '
                "MetricName=\"routing_latency_ms_by_model\"', 'SampleCount', 60)"
            ),
            label="",
            period=Duration.minutes(1),
        )

        dashboard = cloudwatch.Dashboard(
            self,
            "RoutingDashboard",
            dashboard_name=f"routeiq-{env_name}-routing",
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Per-model routing latency (Average, SEARCH expansion)",
                left=[per_model_latency],
                width=24,
                height=6,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Per-model request volume (SampleCount of latency metric)",
                left=[per_model_volume],
                width=24,
                height=6,
            ),
        )
        return dashboard

    def amp_remote_write_statement(self) -> iam.PolicyStatement:
        """Return the ``aps:RemoteWrite`` PolicyStatement for this AMP workspace.

        RouteIQ-74c0/717b. ARN-scoped to ``self.amp_workspace_arn`` (NEVER ``*``).
        Returned as a STATEMENT (not added to a role) so the composition root can
        own the ``iam.Policy`` cross-stack without closing a DependencyCycle (the
        ``add_to_principal_policy`` form mutates the imported role's default policy
        in its OWN stack -- see RouteIqObservabilityStack._grant_pod_role). ASCII sid.
        """
        return iam.PolicyStatement(
            sid="AmpRemoteWrite",
            effect=iam.Effect.ALLOW,
            actions=["aps:RemoteWrite"],
            resources=[self.amp_workspace_arn],
        )

    def amp_remote_write_grant(self, pod_role: iam.IRole) -> None:
        """Grant a pod role ``aps:RemoteWrite`` on this AMP workspace.

        The documented LATE-BINDING seam: when both stacks deploy together, a
        combined ``app.py`` (or operator step) wires the pod-role ``aps:RemoteWrite``
        grant. NOTE: do NOT call this cross-stack -- ``add_to_principal_policy``
        mutates the imported role's default policy in its OWN stack, which closes a
        DependencyCycle when the AMP workspace ARN is owned by another stack. The
        combined-deploy path (RouteIQ-74c0/717b) instead feeds
        :meth:`amp_remote_write_statement` into a composition-root-owned
        ``iam.Policy`` (RouteIqObservabilityStack._grant_pod_role). This helper is
        the single-stack / same-stack convenience wrapper.
        """
        pod_role.add_to_principal_policy(self.amp_remote_write_statement())
