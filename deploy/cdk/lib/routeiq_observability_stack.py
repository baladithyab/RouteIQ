"""RouteIqObservabilityStack - the RouteIQ P2 config-state + observability + lake.

The P2 composition root, a SEPARATE stack from the P0 ``RouteIqStack`` (the
~30-minute-rollback rule: config-state + observability + the data lake roll back
independently of the P0 network/EKS/ECR substrate). It wires the three P2
constructs:

    ConfigStateConstruct   - AppConfig application/environment/profile/strategy +
                             the config-validator Lambda (config_state_construct).
    ObservabilityConstruct - AMP workspace, FLAG-GATED-OFF AMG, the TLS-enforced
                             SNS on-call topic, and the CloudWatch Logs metric
                             filters + alarms over the routing log group, incl.
                             the PER-MODEL dimensioned filter keyed on the OTel
                             ``gen_ai.response.model`` field (observability_construct).
    DataLakeConstruct      - FLAG-GATED-OFF routing_decision CW Logs -> Firehose ->
                             S3 (Parquet) -> Glue/Athena analytics lake
                             (data_lake_construct).

CRED-FREE / PROPS-ONLY P0 COUPLING (the load-bearing boundary):
The metric filters + the data-lake subscription both attach to P0's DEDICATED
routing log group (``/aws/containerinsights/<cluster>/routeiq-routing``,
EksClusterConstruct.routing_log_group_name). A CFN ``AWS::Logs::MetricFilter`` /
``AWS::Logs::SubscriptionFilter`` requires its target group to ALREADY EXIST at
deploy time, but this stack must synth WITHOUT AWS creds (the cred-free gate). So
the group is NOT imported via ``from_lookup`` (which needs creds) - it is either:

  - re-imported by NAME via ``logs.LogGroup.from_log_group_name`` when the
    operator passes ``routing_log_group_name`` (the P0 output, the normal path);
    OR
  - if neither a name nor an ``ILogGroup`` is supplied, derived from the P0
    naming convention via the shared ``lib.naming.routing_log_group_name`` helper
    (RouteIQ-45fa: ONE source of truth) so a standalone synth + the cred-free tests
    proceed. The P0 stack OWNS the group; this stack only references it.

THE COMBINED-DEPLOY SEAM (``foundation=`` -- RouteIQ-569f / 81c4 / 74c0 / 717b):
when ``app.py`` threads the P0 ``RouteIqStack`` by reference, this stack:
  * references the REAL P0 ``ILogGroup`` (cross-stack ``Fn::ImportValue`` for the
    filter ``LogGroupName``) + ``add_dependency(foundation)`` so CFN deploys P0
    before P2 and a MetricFilter CREATE never races a missing group (81c4);
  * owns a P2-stack ``iam.Policy`` ("PodObsGrants") attached to the imported P0
    pod role, carrying the AppConfig runtime-poll grant (569f, the
    ``appconfigdata`` data-plane actions ADR-0026 needs) + the ``aps:RemoteWrite``
    grant (74c0/717b). The policy is OWNED HERE (not via
    ``pod_role.add_to_principal_policy``) so it does not mutate the P0 role's
    default policy cross-stack -- that would close a DependencyCycle (the
    RouteIqStateStack ``PodStateGrants`` pattern).
When ``foundation`` is None (the standalone / separate-pipeline path) every grant
and the dependency are SKIPPED, so the default cred-free synth is byte-identical
and the grants are applied out-of-band (the construct ``*_grant`` seams /
``*_statement`` helpers, see the chart README stack-composition note).

ATHENA WORKGROUP: neither the DataLakeConstruct nor the VSR source provisions an
Athena workgroup (the construct exposes only ``athena_database``). The
operator-facing ``AthenaWorkgroup`` output the P2 mandate requires is created
HERE, in the composition root, ONLY when the lake is enabled, with encrypted
(SSE-KMS, lake CMK) query results landing under an ``athena-results/`` prefix in
the lake bucket - so the stack emits a real, queryable workgroup without
modifying the read-only construct.

ASCII / Latin-1-only Descriptions (P0 section 4.5): an em-dash (U+2014) passes
``cdk synth`` but FAILS the IAM/CFN CREATE API. Every Description here stays ASCII.

CfnOutputs (the operator-visible P2 contract):
    AmpWorkspaceId       - the AMP workspace id (kubectl/ADOT remote-write target)
    AmpRemoteWriteUrl    - the composed AMP remote_write endpoint URL
    AppConfigApplicationId / AppConfigProfileArn - the AppConfig poll contract
    AppConfigArn         - the env-scoped AppConfig configuration ARN
    AlarmTopicArn        - the TLS-enforced on-call SNS topic
    DataLakeBucket / AthenaWorkgroup / AthenaDatabase - lake outputs (lake on only)

``_suppress_nag()`` runs LAST (every resource exists before suppression by path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aws_cdk import CfnOutput, Stack, Tags
from aws_cdk import aws_athena as athena
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from constructs import Construct

from .config_state_construct import ConfigStateConstruct
from .data_lake_construct import DataLakeConstruct
from .naming import routing_log_group_name as _routing_log_group_name
from .obs_nag_suppressions import apply_observability_nag_suppressions
from .observability_construct import ObservabilityConstruct

if TYPE_CHECKING:  # pragma: no cover - forward ref only
    from .routeiq_stack import RouteIqStack


class RouteIqObservabilityStack(Stack):
    """The P2 stack: AppConfig config-state + AMP/AMG/CW observability + data lake."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        # The P0 stack, passed by REFERENCE for cred-free combined-deploy wiring
        # (mirrors RouteIqStateStack.foundation). When given, the P2 stack:
        #   * grants the P0 pod role the AppConfig poll actions (RouteIQ-569f) and
        #     aps:RemoteWrite (RouteIQ-74c0/717b) via a P2-stack-owned iam.Policy
        #     attached to the imported role (NOT pod_role.add_to_principal_policy,
        #     which would mutate the P0 role's default policy = a P0->P2 edge =
        #     DependencyCycle -- the exact state-stack trap);
        #   * references the P0 ILogGroup so CDK emits a cross-stack Fn::ImportValue
        #     for the filter LogGroupName, and add_dependency(foundation) so CFN
        #     deploys P0 before P2 (RouteIQ-81c4: the MetricFilter CREATE can't race
        #     a missing group).
        # Optional so the default flag-off / standalone synth stays byte-identical
        # (every grant + the dependency are guarded ``if foundation is not None``).
        foundation: RouteIqStack | None = None,
        routing_log_group_name: str | None = None,
        enable_amg: bool = False,
        enable_data_lake: bool = False,
        notify_emails: list[str] | None = None,
        cost_center: str | None = None,
        team: str | None = None,
        tenant: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self.enable_amg = enable_amg
        self.enable_data_lake = enable_data_lake

        # Stack tag on every taggable resource (Tags.of propagates). Cost tags are
        # emitted ONLY when supplied so the default synth stays byte-stable for the
        # snapshot test (mirrors the P0 RouteIqStack tagging contract).
        Tags.of(self).add("routeiq:env", env_name)
        for _key, _value in (
            ("routeiq:cost-center", cost_center),
            ("routeiq:team", team),
            ("routeiq:tenant", tenant),
        ):
            if _value:
                Tags.of(self).add(_key, _value)

        # -- Reference the P0 routing log group (props-only, NEVER from_lookup) --
        # RouteIQ-81c4: prefer the REAL cross-stack reference when a combined-deploy
        # app threads the P0 foundation. Reading foundation.eks_cluster.routing_log_group
        # (a real ILogGroup in the SAME cdk.App) makes CDK emit an auto-generated
        # Export / Fn::ImportValue for the group NAME at synth (cred-free, no
        # from_lookup), so the P2 MetricFilter / lake SubscriptionFilter LogGroupName
        # becomes an Fn::ImportValue token AND a stack-level dependency is recorded
        # via add_dependency(foundation) below -- CFN then deploys P0 (the group)
        # before P2 (the filters), so a MetricFilter CREATE can never race a missing
        # group. The by-NAME fallback (operator-supplied name or the P0 naming
        # convention) is KEPT for the standalone / separate-pipeline path so the
        # default cred-free synth stays byte-identical.
        self.foundation = foundation
        if foundation is not None:
            self.routing_log_group = foundation.eks_cluster.routing_log_group
            self.routing_log_group_name = foundation.eks_cluster.routing_log_group_name
            # CFN enforces P0-before-P2 (the MetricFilter CREATE needs the group).
            self.add_dependency(foundation)
        else:
            self.routing_log_group_name = routing_log_group_name or _routing_log_group_name(
                env_name
            )
            self.routing_log_group = logs.LogGroup.from_log_group_name(
                self, "ImportedRoutingLogGroup", self.routing_log_group_name
            )

        # -- 1. AppConfig config-state -----------------------------------------
        self.config_state = ConfigStateConstruct(
            self,
            "ConfigStateConstruct",
            env_name=env_name,
        )

        # -- 2. AMP + flag-gated AMG + TLS SNS + routing filters/alarms ---------
        # Attribute name MUST be ``observability`` - obs_nag_suppressions reads
        # ``stack.observability`` (getattr-guarded) for the AMG-role IAM5 helper.
        self.observability = ObservabilityConstruct(
            self,
            "ObservabilityConstruct",
            env_name=env_name,
            routing_log_group=self.routing_log_group,
            enable_amg=enable_amg,
            notify_emails=notify_emails,
        )

        # -- 3. routing_decision -> Firehose -> S3 Parquet -> Glue/Athena lake --
        # FLAG-GATED off by default so the default synth is byte-identical. The
        # source is the SAME imported P0 routing log group (the lake subscription
        # and the metric filters share one group).
        self.data_lake: DataLakeConstruct | None = None
        self.athena_workgroup: athena.CfnWorkGroup | None = None
        if enable_data_lake:
            self.data_lake = DataLakeConstruct(
                self,
                "DataLakeConstruct",
                env_name=env_name,
                source_log_group=self.routing_log_group,
            )
            self.athena_workgroup = self._build_athena_workgroup(self.data_lake)

        # -- 3b. Cross-stack pod-role grants (the combined-deploy seam) ---------
        # RouteIQ-569f (AppConfig runtime poll) + RouteIQ-74c0/717b (aps:RemoteWrite).
        # Applied ONLY when the P0 foundation is threaded (combined deploy). Guarded
        # so the default flag-off / standalone synth stays byte-identical.
        self.pod_obs_grants: iam.Policy | None = None
        if foundation is not None:
            self.pod_obs_grants = self._grant_pod_role(foundation.pod_role)

        # -- 4. Operator-visible CfnOutputs ------------------------------------
        self._add_outputs()

        # -- 5. cdk-nag suppressions (LAST, by path) ---------------------------
        self._suppress_nag()

    def _grant_pod_role(self, pod_role: iam.IRole) -> iam.Policy:
        """Attach the P2 pod-role grants via a P2-STACK-OWNED iam.Policy.

        RouteIQ-569f + RouteIQ-74c0/717b. The combined-deploy seam: the P0 pod role
        lives in the P0 stack, so it is threaded in by reference (foundation.pod_role)
        and CDK resolves its ARN as a cross-stack Fn::ImportValue.

        CRITICAL CROSS-STACK CYCLE AVOIDANCE (the exact RouteIqStateStack pattern,
        routeiq_state_stack.py:249-293): we do NOT call the construct grant helpers'
        ``pod_role.add_to_principal_policy`` form here. That mutates the imported P0
        role's DEFAULT policy, which is synthesised IN THE P0 STACK -- pushing this
        P2 stack's resource ARNs (the AppConfig profile ARN, the AMP workspace ARN)
        into a P0-stack resource creates a P0 -> P2 dependency, while P2 already
        depends on P0 (the log-group import + add_dependency). That closes a
        DependencyCycle at synth. Instead this stack OWNS a separate ``iam.Policy``
        and ``attach_to_role`` the imported role: the policy + the P2-owned ARNs stay
        in the P2 template, the only cross-stack edge is P2 -> P0 (the role import),
        which already exists. Equivalent runtime grant, no cycle.

        The statements are sourced from the constructs that own the resources
        (ConfigStateConstruct.appconfig_poll_statement / ObservabilityConstruct.
        amp_remote_write_statement) so each ARN scope stays with its owner. Both are
        ARN-scoped (never ``*``), so no new nag suppression is needed on the P0 role.
        """
        policy = iam.Policy(
            self,
            "PodObsGrants",
            statements=[
                # RouteIQ-569f: AppConfig runtime-poll, scoped to the env-scoped
                # profile ARN (NOT ``*``). Carries the appconfigdata data-plane
                # actions the boto3 AppConfigData client uses (the gateway's runtime
                # poll), plus the control-plane GET for SDK-path tolerance.
                self.config_state.appconfig_poll_statement(),
                # RouteIQ-74c0/717b: aps:RemoteWrite scoped to the AMP workspace ARN
                # (the previously-defined-but-never-called amp_remote_write_grant
                # seam, now wired in the combined-deploy path).
                self.observability.amp_remote_write_statement(),
            ],
        )
        policy.attach_to_role(pod_role)
        return policy

    def _build_athena_workgroup(self, data_lake: DataLakeConstruct) -> athena.CfnWorkGroup:
        """Create the operator-facing Athena workgroup for the routing-decisions lake.

        Query results land under an ``athena-results/`` prefix in the lake bucket,
        SSE-KMS-encrypted with the lake CMK (satisfies AwsSolutions-ATH1 - encrypted
        query results). ``enforce_workgroup_configuration`` so a console user cannot
        override the encryption/output settings. The DataLakeConstruct itself does
        not provision a workgroup (parity with the VSR source), so it is composed
        here only when the lake is enabled.
        """
        result_location = f"s3://{data_lake.bucket.bucket_name}/athena-results/"
        return athena.CfnWorkGroup(
            self,
            "AthenaWorkGroup",
            name=f"routeiq-{self.env_name}-routing-decisions",
            description=(
                "RouteIQ routing-decisions analytics workgroup (Glue/Athena over "
                "the Parquet lake). Encrypted query results in the lake bucket."
            ),
            recursive_delete_option=True,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                enforce_work_group_configuration=True,
                publish_cloud_watch_metrics_enabled=True,
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=result_location,
                    encryption_configuration=athena.CfnWorkGroup.EncryptionConfigurationProperty(
                        encryption_option="SSE_KMS",
                        kms_key=data_lake.kms_key.key_arn,
                    ),
                ),
            ),
        )

    def _add_outputs(self) -> None:
        """Emit the operator-visible P2 CfnOutputs."""
        obs = self.observability
        cfg = self.config_state

        CfnOutput(
            self,
            "AmpWorkspaceId",
            value=obs.amp_workspace.attr_workspace_id,
            description="Amazon Managed Prometheus workspace id (ADOT remote-write target)",
        )
        CfnOutput(
            self,
            "AmpRemoteWriteUrl",
            value=obs.amp_remote_write_url,
            description="AMP remote_write endpoint URL for the pod ADOT exporter",
        )
        CfnOutput(
            self,
            "AlarmTopicArn",
            value=obs.alarm_topic.topic_arn,
            description="TLS-enforced on-call SNS topic every routing alarm pages",
        )
        CfnOutput(
            self,
            "AppConfigApplicationId",
            value=cfg.appconfig_application_id,
            description="AppConfig application id (routeiq) for the config poll path",
        )
        CfnOutput(
            self,
            "AppConfigProfileArn",
            value=cfg.appconfig_profile_arn,
            description=(
                "AppConfig env-scoped configuration ARN the gateway polls via "
                "StartConfigurationSession + GetLatestConfiguration"
            ),
        )
        # The P2 mandate names this output ``AppConfigArn``; it is the same
        # env-scoped configuration ARN, exported under the mandated name too.
        CfnOutput(
            self,
            "AppConfigArn",
            value=cfg.appconfig_profile_arn,
            description="AppConfig configuration ARN (alias of AppConfigProfileArn)",
        )

        # AMG output only when the workspace exists (flag-gated).
        if obs.amg_workspace is not None:
            CfnOutput(
                self,
                "AmgWorkspaceEndpoint",
                value=obs.amg_workspace.attr_endpoint,
                description="Amazon Managed Grafana workspace endpoint (enable_amg=true)",
            )

        # Lake outputs only when the lake exists (flag-gated).
        if self.data_lake is not None:
            CfnOutput(
                self,
                "DataLakeBucket",
                value=self.data_lake.bucket.bucket_name,
                description="routing-decisions Parquet data-lake S3 bucket",
            )
            CfnOutput(
                self,
                "AthenaDatabase",
                value=self.data_lake.athena_database,
                description="Glue/Athena database for the routing-decisions table",
            )
        if self.athena_workgroup is not None:
            CfnOutput(
                self,
                "AthenaWorkgroup",
                value=self.athena_workgroup.name,
                description="Athena workgroup for routing-decisions analytics (encrypted results)",
            )

    def _suppress_nag(self) -> None:
        """Apply the evidenced cdk-nag suppressions for this stack (LAST).

        The ObservabilityConstruct's only finding is the AMG data-source role's
        read-scope wildcards (flag-gated, getattr-guarded). The ConfigStateConstruct
        + DataLakeConstruct apply their own INLINE suppressions at construction
        time. So the only stack-level fan-out is the observability helper.
        """
        apply_observability_nag_suppressions(self)
