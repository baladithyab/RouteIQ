"""RouteIqStack - the single P0 composition root for RouteIQ on AWS.

ONE stack (proposal P0 doc 31 section 6): a private multi-AZ EKS Auto Mode
cluster, an ECR GHCR pull-through cache, ONE least-privilege pod IAM role bound
via EKS Pod Identity, and the network substrate they all sit on. P1 (Aurora +
ElastiCache) is a SEPARATE stack/CI-stage per the ~30-minute-rollback rule, so
P0 ships exactly one RouteIqStack.

This mirrors the vllm-sr-on-aws ``vllm_sr_eks_stack.py`` composition root,
re-derived symbol-by-symbol from the real source, with the RouteIQ divergences:

- **Pod Identity, not IRSA** (proposal section 3). The pod->role binding is a
  single ``eks.CfnPodIdentityAssociation`` over a STATIC
  ``pods.eks.amazonaws.com`` trust. There is NO ``OpenIdConnectProvider``, NO
  ``CfnJson`` trust map, NO ``WebIdentityPrincipal``. The role's trust is a
  service-principal grant of ``sts:AssumeRole`` + ``sts:TagSession`` to
  ``pods.eks.amazonaws.com``.
- **ONE pod role** (proposal section 4). RouteIQ is a single stateless gateway
  pod, so P0 mints ONE role bound to ONE ``(namespace, serviceAccount)``. The
  VSR router / EAIG / bearer-minter three-role split and the cross-account
  capacity loop are dropped entirely.
- **SINGLE-ACCOUNT** (operator-confirmed): one static pod role in the deploy
  account, no cross-account ``targetRoleArn``.

Wiring order (proposal section 10):

    1. NetworkConstruct
    2. EksClusterConstruct(vpc=...)  + the defensive pod-identity-agent add-on
    3. EcrConstruct  (GHCR pull-through cache surface)
    4. the ONE pod role + pod_identity_association(namespace, service_account)
    5. CfnOutputs
    6. self._suppress_nag()  (LAST, so every resource exists before suppression
       is applied by path)

Every IAM role/statement Description is ASCII / Latin-1 only (proposal section
4.5): an em-dash (U+2014) passes ``cdk synth`` but FAILS the IAM CREATE API.
"""

from __future__ import annotations

from aws_cdk import Aws, CfnOutput, CfnParameter, Stack, Tags
from aws_cdk import aws_iam as iam
from constructs import Construct

from .ecr_construct import EcrConstruct
from .eks_cluster_construct import EksClusterConstruct
from .nag_suppressions import apply_nag_suppressions
from .naming import routing_log_group_export_name
from .network_construct import NetworkConstruct
from .waf_construct import WafConstruct


class RouteIqStack(Stack):
    """The single P0 stack: network + EKS Auto Mode + ECR GHCR + pod IAM role."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        vpc_cidr: str = "10.40.0.0/16",
        nat_gateways: int = 1,
        k8s_version: str = "1.33",
        enable_ghcr_ptc: bool = True,
        sa_namespace: str = "routeiq",
        sa_name: str = "routeiq-gateway",
        image_tag: str = "1.0.0-rc1",
        admin_principal_arns: list[str] | None = None,
        bedrock_model_arns: list[str] | None = None,
        capacity_account_ids: list[str] | None = None,
        config_s3_bucket: str | None = None,
        secret_arns: list[str] | None = None,
        enable_waf: bool = False,
        waf_alb_arn: str | None = None,
        waf_rate_limit: int | None = None,
        waf_crs_block: bool = False,
        waf_rate_block: bool = False,
        cost_center: str | None = None,
        team: str | None = None,
        tenant: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self.sa_namespace = sa_namespace
        self.sa_name = sa_name
        self.image_tag = image_tag
        self._admin_principal_arns = list(admin_principal_arns or [])
        self._bedrock_model_arns = list(bedrock_model_arns or [])
        self._capacity_account_ids = list(capacity_account_ids or [])
        self._config_s3_bucket = config_s3_bucket
        self._secret_arns = list(secret_arns or [])
        self._enable_waf = bool(enable_waf)
        self._waf_alb_arn = waf_alb_arn
        self._waf_rate_limit = waf_rate_limit
        self._waf_crs_block = bool(waf_crs_block)
        self._waf_rate_block = bool(waf_rate_block)

        # Stack tag on every taggable resource (Tags.of propagates). Cost tags
        # are emitted ONLY when supplied so the default synth stays byte-stable
        # for the snapshot test (proposal section 10).
        Tags.of(self).add("routeiq:env", env_name)
        for _key, _value in (
            ("routeiq:cost-center", cost_center),
            ("routeiq:team", team),
            ("routeiq:tenant", tenant),
        ):
            if _value:
                Tags.of(self).add(_key, _value)

        # -- 1. Network substrate ---------------------------------------------
        # VPC + three SGs (alb / pod / vpce) + five interface endpoints
        # (incl. BEDROCK_RUNTIME) + the S3 gateway endpoint. The gateway pod ENIs
        # live in the private-app (PRIVATE_WITH_EGRESS) tier.
        self.network = NetworkConstruct(
            self,
            "NetworkConstruct",
            env_name=env_name,
            vpc_cidr=vpc_cidr,
            nat_gateways=nat_gateways,
        )

        # -- 2. EKS Auto Mode cluster -----------------------------------------
        # L1 CfnCluster (the three Auto Mode blocks) + the two hand-built Auto
        # Mode IAM roles (cluster + node) + node/admin CfnAccessEntry resources +
        # the defensive eks-pod-identity-agent CfnAddon. The cluster control-plane
        # ENIs + Auto Mode nodes land in the private-app subnets.
        self.eks_cluster = EksClusterConstruct(
            self,
            "EksClusterConstruct",
            env_name=env_name,
            cluster_subnets=self.network.private_app_subnets,
            version=k8s_version,
            admin_principal_arns=self._admin_principal_arns,
        )
        # Container insights: ships the amazon-cloudwatch-observability add-on +
        # the CDK-created RoutingLogGroup at P0 (proposal section 7.5). The
        # dimensioned RoutingLatencyByModel MetricFilter is PREP-ONLY / deferred
        # (data-source-blocked) and flag-gated off inside the construct.
        self.eks_cluster.enable_container_insights("ContainerInsights")

        # -- 3. ECR GHCR pull-through cache surface ---------------------------
        # The credential secret ARN is a deploy-time CfnParameter, NEVER a literal
        # (proposal section 9.2): it points at an operator-provisioned,
        # ``ecr-pullthroughcache/``-prefixed Secrets Manager secret holding a real
        # GitHub PAT. CDK does NOT create the secret. A CfnParameter keeps it out
        # of source and out of the snapshot baseline while still feeding the
        # CfnPullThroughCacheRule.credentialArn. (NoEcho so the param value is not
        # echoed in the CloudFormation console.)
        self.ghcr_credential_arn_param = CfnParameter(
            self,
            "GhcrCredentialSecretArn",
            type="String",
            default="",
            no_echo=True,
            description=(
                "ARN of the operator-provisioned ecr-pullthroughcache/ Secrets "
                "Manager secret holding the GHCR PAT. NOT created by CDK."
            ),
        )
        self.ecr = EcrConstruct(
            self,
            "EcrConstruct",
            credential_arn=self.ghcr_credential_arn_param.value_as_string,
            enable_ghcr_ptc=enable_ghcr_ptc,
        )

        # -- 4. The ONE pod IAM role + the Pod Identity association ------------
        # Trust: a STATIC service-principal grant to pods.eks.amazonaws.com of
        # sts:AssumeRole + sts:TagSession (proposal section 4.1). There is NO
        # WebIdentityPrincipal, NO OpenIdConnectPrincipal, NO CfnJson condition
        # map. Pod Identity creds come from the pod-identity agent, NOT an STS
        # web-identity exchange, so sts:AssumeRoleWithWebIdentity is NOT granted.
        # ASCII-only description (proposal section 4.5).
        self.pod_role = iam.Role(
            self,
            "PodRole",
            assumed_by=iam.ServicePrincipal("pods.eks.amazonaws.com"),
            description=f"RouteIQ {env_name} gateway pod role (EKS Pod Identity)",
        )
        # Pod Identity requires sts:TagSession alongside the implicit
        # sts:AssumeRole on the trust document.
        assert self.pod_role.assume_role_policy is not None
        self.pod_role.assume_role_policy.add_statements(
            iam.PolicyStatement(
                actions=["sts:TagSession"],
                principals=[iam.ServicePrincipal("pods.eks.amazonaws.com")],
            )
        )

        self._add_pod_role_statements()
        self._add_capacity_assume_grant()

        # The pod->role binding. Keyed on (namespace, serviceAccount) matching the
        # chart's rendered ServiceAccount; NO eks.amazonaws.com/role-arn annotation
        # is needed on the chart side (proposal section 3.5 / 11.2). The helper
        # makes the association DependsOn the defensive eks-pod-identity-agent
        # add-on so it is guaranteed to resolve.
        self.pod_association = self.eks_cluster.pod_identity_association(
            "PodIdentityAssociation",
            namespace=sa_namespace,
            service_account=sa_name,
            role=self.pod_role,
        )

        # -- 5. CfnOutputs the chart / CI consume -----------------------------
        self._add_outputs()

        # -- 5b. WAFv2 edge layer (RouteIQ-4f59; flag-gated, DEFAULT OFF) ------
        # Instantiated ONLY when routeiq:enable_waf is True AND an operator
        # supplies a waf_alb_arn. The `and self._waf_alb_arn` guard is
        # load-bearing: at P0 the chart default is service.type=ClusterIP /
        # ingress.enabled=false, so EKS Auto Mode renders NO ALB - there is no
        # in-stack ALB ARN to associate to. The construct + its synth test (the
        # association to a SUPPLIED ARN) are the cred-free deliverables; the LIVE
        # attach is operator-gated (flip ingress.enabled=true -> Auto Mode renders
        # an ALB -> pass its ARN as routeiq:waf_alb_arn). With the flag OFF (the
        # default surface) NO WafConstruct is built, so the stack emits ZERO
        # AWS::WAFv2::* resources and test_p4_edge_no_cdk_alb + the snapshot stay
        # byte-stable. RouteIQ-4f59 closes PARTIAL (live attach deploy-gated).
        self.waf: WafConstruct | None = None
        if self._enable_waf and self._waf_alb_arn:
            self.waf = WafConstruct(
                self,
                "WafConstruct",
                env_name=env_name,
                alb_arn=self._waf_alb_arn,
                rate_limit=self._waf_rate_limit,
                crs_block=self._waf_crs_block,
                rate_block=self._waf_rate_block,
            )

        # -- 6. cdk-nag suppressions (LAST, by path) --------------------------
        self._suppress_nag()

    def _add_pod_role_statements(self) -> None:
        """Attach the P0 least-privilege pod-role statements.

        Each statement is its own ``PolicyStatement`` with an explicit ``sid``,
        least-privilege (explicit resources, never ``*`` except where the action
        itself requires it). The grant set is exactly Bedrock invoke + Secrets +
        config S3 + Logs (proposal section 4.2). NO ``sts:AssumeRoleWithWebIdentity``
        (Pod Identity does not use it), NO ``ecr:*`` (ECR pull lives on the node
        role - proposal section 4.4).
        """
        # BedrockInvoke (P0, data-plane critical). Scoped to the supplied model
        # ARNs when given; otherwise the region foundation-model wildcard - the
        # only valid form for a router whose model set is deploy-time dynamic
        # (proposal section 4.2; suppressed under AwsSolutions-IAM5).
        bedrock_resources = self._bedrock_model_arns or [
            f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}::foundation-model/*"
        ]
        self.pod_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvoke",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
                ],
                resources=bedrock_resources,
            )
        )

        # SecretsRead (P0): the LITELLM_MASTER_KEY / ADMIN_API_KEYS / provider-key
        # secret ARNs. Scoped to the supplied secret ARNs; defaults to the
        # account/region routeiq secret-name wildcard so the role is deployable
        # before the operator pins exact ARNs.
        secret_resources = self._secret_arns or [
            f"arn:{Aws.PARTITION}:secretsmanager:{Aws.REGION}:{Aws.ACCOUNT_ID}:secret:routeiq/*"
        ]
        self.pod_role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsRead",
                actions=[
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                ],
                resources=secret_resources,
            )
        )

        # ConfigS3Read (P0): the config bucket object reads (GetObjectAttributes
        # is the ETag poll the gateway.configSync.s3 path uses). Only added when a
        # config bucket is supplied - omitting it keeps the role narrow and the
        # default synth byte-stable.
        if self._config_s3_bucket:
            self.pod_role.add_to_policy(
                iam.PolicyStatement(
                    sid="ConfigS3Read",
                    actions=["s3:GetObject", "s3:GetObjectAttributes"],
                    resources=[f"arn:{Aws.PARTITION}:s3:::{self._config_s3_bucket}/*"],
                )
            )

        # Logs (P0): the CDK-created routing log group ONLY (NOT logs:* - Auto
        # Mode + the amazon-cloudwatch-observability add-on largely handle pod
        # logs). The MetricFilters consume this group (proposal section 4.2 / 7.5).
        log_group_name = getattr(self.eks_cluster, "routing_log_group_name", None)
        if log_group_name is not None:
            self.pod_role.add_to_policy(
                iam.PolicyStatement(
                    sid="Logs",
                    actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                    resources=[
                        f"arn:{Aws.PARTITION}:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:"
                        f"log-group:{log_group_name}:*"
                    ],
                )
            )

        # --- Forward-compatible statements, resource-pending until peer P-tiers
        # land (proposal section 4.3). Kept as documentation, NOT emitted at P0,
        # so the role stays least-privilege and the snapshot stays byte-stable:
        #   P2 (ADR-0026): appconfig:GetLatestConfiguration +
        #                  appconfigdata:StartConfigurationSession (AppConfig ARN)
        #   P2 (ADR-0027): aps:RemoteWrite (AMP workspace ARN)
        #   P1 (ADR-0028): rds-db:connect (Aurora dbuser ARN)
        #   P1 (ADR-0029): elasticache:Connect (cache + IAM-user ARN)

    def _add_capacity_assume_grant(self) -> None:
        """RouteIQ-6150 (C1): grant the pod role sts:AssumeRole on cross-account
        Bedrock capacity member roles (flag-gated; empty list => byte-stable).

        For each account id in ``routeiq:capacity_account_ids`` this computes the
        PREDICTABLE member-role ARN
        ``arn:<part>:iam::<acct>:role/RouteIqBedrockCapacity-<env>`` (the stable
        name BedrockCapacityMemberStack mints in that account) and grants the home
        gateway pod role plain ``sts:AssumeRole`` on EXACTLY those ARNs - never a
        wildcard, so AwsSolutions-IAM5 does NOT fire and no new suppression is
        needed.

        This is the RouteIQ DIVERGENCE from VSR: VSR put the home grant on a
        separate bearer-minter role (its native path used web-identity and needed
        no home assume); RouteIQ has no minter and no web-identity, so the grant
        lands on the single Pod-Identity pod role directly.

        Each member ARN becomes one LiteLLM ``model_list`` row's
        ``litellm_params.aws_role_name`` (doc 50 section 2c): LiteLLM's BaseAWSLLM
        STS-assumes it (the assume this statement authorizes) to borrow that
        account's Bedrock quota. The per-account CfnOutput surfaces the ARN the
        operator pastes into that row.

        Empty ``capacity_account_ids`` (the default) => zero statements, zero
        outputs => the default synth is byte-identical (the snapshot stays green).
        """
        if not self._capacity_account_ids:
            return

        member_role_arns = [
            f"arn:{Aws.PARTITION}:iam::{acct}:role/RouteIqBedrockCapacity-{self.env_name}"
            for acct in self._capacity_account_ids
        ]
        self.pod_role.add_to_policy(
            iam.PolicyStatement(
                sid="AssumeCapacityRoles",
                actions=["sts:AssumeRole"],
                resources=member_role_arns,  # explicit ARNs, NOT a wildcard
            )
        )
        for acct, arn in zip(self._capacity_account_ids, member_role_arns, strict=True):
            CfnOutput(
                self,
                f"CapacityRoleArn{acct}",
                value=arn,
                description=(
                    f"Bedrock capacity role ARN in account {acct} (deploy "
                    "BedrockCapacityMemberStack there). Set as "
                    "litellm_params.aws_role_name for that account model_list row."
                ),
            )

    def _add_outputs(self) -> None:
        """Emit the operator-visible CfnOutputs (proposal section 7.6 / 10).

        The IRSA-only ``OidcProviderArn`` / ``OidcProviderIssuerUrl`` outputs are
        DROPPED - there is no OIDC provider on the Pod Identity path.
        ``PodAssociationId`` is the new operator-visible binding output.
        """
        CfnOutput(
            self,
            "ClusterName",
            value=self.eks_cluster.cluster_name,
            description="EKS cluster name for aws eks update-kubeconfig + helm upgrade",
        )
        CfnOutput(
            self,
            "ClusterEndpoint",
            value=self.eks_cluster.cluster_endpoint,
            description="EKS API server endpoint",
        )
        CfnOutput(
            self,
            "PodRoleArn",
            value=self.pod_role.role_arn,
            description="The single gateway pod role bound via EKS Pod Identity",
        )
        CfnOutput(
            self,
            "PodAssociationId",
            value=self.pod_association.attr_association_id,
            description=(
                f"Pod Identity association id (binds {self.sa_namespace}/"
                f"{self.sa_name} to PodRoleArn)"
            ),
        )
        CfnOutput(
            self,
            "NodeRoleName",
            value=self.eks_cluster.node_role.role_name,
            description="EKS Auto Mode node role NAME",
        )
        CfnOutput(
            self,
            "EcrGhcrPrefix",
            value=self.ecr.ghcr_prefix_value,
            description=(
                "chart image.repository override base "
                "(append /baladithyab/routeiq at helm upgrade time)"
            ),
        )
        # RouteIQ-81c4: a STABLE cross-stack EXPORT of the routing log group name so
        # the P2 RouteIqObservabilityStack can Fn::ImportValue it (instead of relying
        # on a hand-copied NAME-only convention). The export name is deterministic
        # (RouteIqStack-<env>-RoutingLogGroupName); once P2 imports it, it cannot be
        # renamed/removed without first removing the P2 consumer
        # (cfn-cross-stack-export-revisioned-ref-deadlock). The combined-deploy P2
        # path actually references the ILogGroup directly (CDK auto-generates its own
        # import), so this named export is the operator-visible + separate-pipeline
        # handoff contract.
        log_group_name = getattr(self.eks_cluster, "routing_log_group_name", None)
        if log_group_name is not None:
            CfnOutput(
                self,
                "RoutingLogGroupName",
                value=log_group_name,
                export_name=routing_log_group_export_name(self.env_name),
                description=(
                    "Dedicated routing-decision CloudWatch log group name. P2 "
                    "observability + the data lake attach metric/subscription "
                    "filters to this group (P0 owns it)."
                ),
            )

    def _suppress_nag(self) -> None:
        """Apply the evidenced cdk-nag suppressions for this stack (LAST)."""
        apply_nag_suppressions(self)
