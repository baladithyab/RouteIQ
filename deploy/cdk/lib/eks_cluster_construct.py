"""EKS Auto Mode cluster for RouteIQ (CDK-native, L1 CfnCluster + Pod Identity).

WHY this shape (P0 foundation; see
docs/architecture/aws-rearchitecture/31-p0-cdk-foundation-proposal.md sections 3 + 7):

EKS **Auto Mode** makes AWS manage the entire data plane - node lifecycle, a
managed Karpenter, AMI/driver selection, autoscaling, load-balancer + EBS
provisioning. That collapses the only part of an ai-on-eks-style stack RouteIQ
would otherwise hand-translate (the Karpenter NodePool/EC2NodeClass templates +
core node group) down to nothing, so RouteIQ provisions a MINIMAL Auto Mode
cluster natively in CDK.

Auto Mode is NOT on the stable L2 ``aws_eks.Cluster`` in a form RouteIQ wants, so
this construct uses the **L1 ``CfnCluster`` escape hatch** from stable
``aws-cdk-lib`` - which exposes ``compute_config{enabled,node_pools,
node_role_arn}`` + ``storage_config.block_storage{enabled}`` +
``kubernetes_network_config.elastic_load_balancing{enabled}``. Those three
blocks toggled together = Auto Mode. This is a PORT of
vllm-sr-on-aws/cdk/lib/eks_cluster_construct.py, re-derived symbol-by-symbol from
the real source, with two divergences called out below (Pod Identity, single
pod role).

**POD IDENTITY, NOT IRSA** (proposal section 3; research verdict, very high
confidence). RouteIQ grants pod IAM via EKS Pod Identity - a single
``eks.CfnPodIdentityAssociation`` over a STATIC ``pods.eks.amazonaws.com`` trust.
There is **NO ``OpenIdConnectProvider``, NO ``CfnJson`` trust map, NO
``oidc_provider_issuer`` derivation, NO ``WebIdentityPrincipal``**. The L1
constraint is the clincher: on L1 there is no ``cluster.addServiceAccount()`` to
hide IRSA's complexity, so the gap between trivial Pod Identity and hand-rolled
IRSA is at its widest. This deletes the entire token-keyed-condition /
``.replace("https://")``-on-a-token class of CDK failure from the build. The VSR
source has the OIDC provider + ``oidc_provider_issuer`` + ``irsa_role()`` factory
with the ``CfnJson`` trust map; the RouteIQ port DELETES all three.

A defensive ``eks-pod-identity-agent`` ``CfnAddon`` is emitted regardless of the
"pre-installed on Auto Mode" claim (proposal section 7.4a): AWS docs say the
agent is built into Auto Mode, but the production VSR construct this ports
installs it by hand, so RouteIQ adds it idempotently
(``resolve_conflicts="OVERWRITE"``) so the association is guaranteed to resolve.

App-layer (the RouteIQ Helm chart / manifests) is deployed via kubectl/helm,
NOT via CDK ``KubernetesManifest``/``HelmChart``. This construct provisions ONLY
the cluster + its two IAM roles + the Pod Identity wiring; everything
Kubernetes-side is applied out-of-band.

SINGLE-ACCOUNT (operator-confirmed): one static pod role in the deploy account,
no cross-account ``targetRoleArn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aws_cdk import Aws, CfnOutput, RemovalPolicy, Tags
from aws_cdk import aws_eks as eks
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from constructs import Construct

if TYPE_CHECKING:  # pragma: no cover - forward refs only
    from aws_cdk import aws_ec2 as ec2

# EKS version for the cluster. Auto Mode tracks AMIs itself; this pins the
# control-plane minor (proposal section 7.1).
_EKS_VERSION = "1.33"

# Auto Mode's two built-in node pools. "general-purpose" runs workloads,
# "system" runs cluster-critical add-ons. Both are AWS-managed - RouteIQ does
# NOT author NodePool/EC2NodeClass CRs for them (that is the point of Auto Mode).
_AUTO_MODE_NODE_POOLS = ["general-purpose", "system"]

# PREP-ONLY at P0 (proposal section 7.5). The dimensioned per-model routing
# latency MetricFilter is data-source-blocked until P2: its source, the
# structured routing_decision CloudWatch JSON log line, is a P2 BUILD-NEW item
# (today the gateway emits an OTel/logger.info event, not the CW JSON line), so
# the filter would match zero events at P0. Flip to True only once the P2
# structured log line ships AND its log group is wired to this group.
_ENABLE_ROUTING_LATENCY_BY_MODEL = False

# The MetricFilter JSON dimension key for the per-model latency filter.
# CRITICAL (ADR-0027:64-67): RouteIQ does NOT emit ``selected_model`` as a
# structured field - the telemetry contract emits ``gen_ai.response.model``
# (telemetry_contracts.py:673). A filter keyed on ``$.selected_model`` matches
# ZERO RouteIQ events. The VSR source uses ``$.selected_model``; the RouteIQ
# port MUST use the telemetry-contract key below.
_ROUTING_MODEL_DIM_KEY = '$.["gen_ai.response.model"]'


class EksClusterConstruct(Construct):
    """Minimal EKS Auto Mode cluster (L1 CfnCluster) + IAM roles + Pod Identity.

    Provisions the L1 ``eks.CfnCluster`` with the three Auto Mode blocks, the
    two hand-built Auto Mode IAM roles (cluster + node, logical ids
    ``ClusterRole`` / ``NodeRole`` re-derived from the VSR source), the node +
    admin ``CfnAccessEntry`` resources, and a defensive
    ``eks-pod-identity-agent`` add-on. Pod IAM is wired via
    :meth:`pod_identity_association` (a single ``eks.CfnPodIdentityAssociation``);
    container insights via :meth:`enable_container_insights`.

    Public attributes for the composition root / outputs:
    ``cluster`` (the L1 ``CfnCluster``), ``cluster_name``, ``cluster_endpoint``,
    ``cluster_arn``, ``cluster_role``, ``node_role``, ``pod_identity_addon``.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        cluster_subnets: list[ec2.ISubnet],
        version: str = _EKS_VERSION,
        admin_principal_arns: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self.cluster_name = f"routeiq-{env_name}"
        self._admin_principal_arns = list(admin_principal_arns or [])

        # -- 1. Cluster IAM role (the EKS control plane assumes this) ----------
        # Auto Mode requires the standard cluster policy PLUS compute/storage/
        # networking/block-storage/load-balancing policies so the AWS-managed
        # data plane can provision nodes, EBS volumes, and ELBs on our behalf.
        # NOTE (proposal section 4.5): every IAM Description is ASCII/Latin-1
        # only - an em-dash (U+2014) passes ``cdk synth`` but FAILS the IAM
        # CREATE API. Logical id "ClusterRole" matches the VSR source (~:109).
        self.cluster_role = iam.Role(
            self,
            "ClusterRole",
            assumed_by=iam.ServicePrincipal("eks.amazonaws.com"),
            description=f"RouteIQ {env_name} EKS Auto Mode cluster role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSClusterPolicy"),
                # Auto Mode control-plane policies (AWS-managed):
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSComputePolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSBlockStoragePolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSLoadBalancingPolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSNetworkingPolicy"),
            ],
        )
        # EKS requires the cluster role to allow the eks.amazonaws.com service
        # principal for Auto Mode (sts:TagSession on AssumeRole). The assert
        # documents the invariant (a Role built with assumed_by always has a
        # trust document) and keeps mypy happy (the attr is typed Optional).
        assert self.cluster_role.assume_role_policy is not None
        self.cluster_role.assume_role_policy.add_statements(
            iam.PolicyStatement(
                actions=["sts:TagSession"],
                principals=[iam.ServicePrincipal("eks.amazonaws.com")],
            )
        )

        # -- 2. Node IAM role (Auto-Mode-managed nodes assume this) -----------
        # Auto Mode nodes need the worker + ECR-pull policies. Pod-level AWS
        # access (Bedrock/Secrets/S3) is granted SEPARATELY via Pod Identity on
        # the application pod role, NOT here - this role is the kubelet/node
        # identity only. ECR *pull* lives here (proposal section 4.4 / 7.2);
        # the pull-through-cache import grant is the node identity's grant, not
        # the application pod role's. Logical id "NodeRole" matches VSR (~:136).
        self.node_role = iam.Role(
            self,
            "NodeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description=f"RouteIQ {env_name} EKS Auto Mode node role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSWorkerNodeMinimalPolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryPullOnly"
                ),
            ],
        )

        # -- 3. The cluster (L1 CfnCluster, Auto Mode) ------------------------
        # VSR shape: derive subnet ID strings from the passed ISubnet list
        # (eks_cluster_construct.py:150 ``subnet_ids = [s.subnet_id for s in
        # cluster_subnets]``).
        subnet_ids = [s.subnet_id for s in cluster_subnets]
        self.cfn_cluster = eks.CfnCluster(
            self,
            "Cluster",
            name=self.cluster_name,
            version=version,
            role_arn=self.cluster_role.role_arn,
            resources_vpc_config=eks.CfnCluster.ResourcesVpcConfigProperty(
                subnet_ids=subnet_ids,
                # Private access is ALWAYS on so in-VPC nodes/workloads reach
                # the API privately. Public access is OFF at P0 (the proposal's
                # private posture; the public edge is a deliberate P-later
                # decision, section 11.1a). kubectl reaches the API in-VPC.
                endpoint_private_access=True,
                endpoint_public_access=False,
            ),
            # Auto Mode = these three blocks enabled together (section 7.1):
            compute_config=eks.CfnCluster.ComputeConfigProperty(
                enabled=True,
                node_pools=_AUTO_MODE_NODE_POOLS,
                node_role_arn=self.node_role.role_arn,
            ),
            storage_config=eks.CfnCluster.StorageConfigProperty(
                block_storage=eks.CfnCluster.BlockStorageProperty(enabled=True),
            ),
            kubernetes_network_config=eks.CfnCluster.KubernetesNetworkConfigProperty(
                elastic_load_balancing=eks.CfnCluster.ElasticLoadBalancingProperty(
                    enabled=True,
                ),
            ),
            # API access entries (the modern replacement for the aws-auth
            # ConfigMap). Grant the cluster creator admin so the deploy identity
            # can run kubectl immediately; additional principals are added as
            # CfnAccessEntry resources below.
            access_config=eks.CfnCluster.AccessConfigProperty(
                authentication_mode="API",
                bootstrap_cluster_creator_admin_permissions=True,
            ),
            # Auto Mode supplies the core add-ons (CoreDNS, kube-proxy, VPC-CNI)
            # itself - do NOT also install the self-managed defaults.
            bootstrap_self_managed_addons=False,
        )
        Tags.of(self.cfn_cluster).add("routeiq:env", env_name)
        Tags.of(self.cfn_cluster).add("routeiq:substrate", "eks-automode")

        # Alias exposed for the composition root / nag-suppression-by-path.
        self.cluster = self.cfn_cluster

        # -- 4. Access entries (CfnAccessEntry) -------------------------------
        # (a) The Auto Mode NODE role - type "EC2", with NO access policies. An
        # AWS EKS API constraint forbids access policies on EC2-type entries
        # (proposal section 7.3a; research report section 2 / src [4]).
        node_entry = eks.CfnAccessEntry(
            self,
            "NodeAccessEntry",
            cluster_name=self.cluster_name,
            principal_arn=self.node_role.role_arn,
            type="EC2",
        )
        # The access entry targets the cluster by NAME; CFN does not infer the
        # dependency from the string - order it after the cluster exists.
        node_entry.add_dependency(self.cfn_cluster)

        # (b) The operator / CI kubectl identities.
        # bootstrap_cluster_creator_admin_permissions covers ONLY the CFN exec
        # role, NOT a human's or the CI role's kubectl identity, so each needs an
        # explicit STANDARD entry + an AmazonEKSClusterAdminPolicy association
        # (proposal section 7.3b). Driven by routeiq:admin_principal_arns. The
        # cluster-admin access scope matches the VSR add_admin_access_entry
        # posture (eks_cluster_construct.py:326-344).
        for index, principal_arn in enumerate(self._admin_principal_arns):
            admin_entry = eks.CfnAccessEntry(
                self,
                f"AdminAccessEntry{index}",
                cluster_name=self.cluster_name,
                principal_arn=principal_arn,
                type="STANDARD",
                access_policies=[
                    eks.CfnAccessEntry.AccessPolicyProperty(
                        policy_arn=(
                            f"arn:{Aws.PARTITION}:eks::aws:cluster-access-policy/"
                            "AmazonEKSClusterAdminPolicy"
                        ),
                        access_scope=eks.CfnAccessEntry.AccessScopeProperty(type="cluster"),
                    )
                ],
            )
            admin_entry.add_dependency(self.cfn_cluster)

        # -- 5. Defensive eks-pod-identity-agent add-on (section 7.4a) --------
        # AWS docs claim the Pod Identity agent is built into Auto Mode, but the
        # production VSR Auto Mode construct installs it by hand
        # (eks_cluster_construct.py:391-398). Add it explicitly and idempotently
        # so a CfnPodIdentityAssociation is guaranteed to resolve regardless of
        # which behavior is live on the target cluster.
        # resolve_conflicts="OVERWRITE" makes re-applying a built-in add-on a
        # harmless no-op. The association (section 7.4) depends on THIS add-on so
        # it never races the agent.
        self.pod_identity_addon = eks.CfnAddon(
            self,
            "PodIdentityAddon",
            addon_name="eks-pod-identity-agent",
            cluster_name=self.cluster_name,
            resolve_conflicts="OVERWRITE",
        )
        self.pod_identity_addon.add_dependency(self.cfn_cluster)

        # -- public attrs for the composition root + outputs ------------------
        self.cluster_endpoint: str = self.cfn_cluster.attr_endpoint
        self.cluster_arn: str = self.cfn_cluster.attr_arn

        CfnOutput(self, "ClusterName", value=self.cluster_name)
        CfnOutput(self, "ClusterEndpoint", value=self.cluster_endpoint)
        CfnOutput(self, "NodeRoleName", value=self.node_role.role_name)
        CfnOutput(
            self,
            "KubectlConfigCommand",
            value=(f"aws eks update-kubeconfig --name {self.cluster_name} --region {Aws.REGION}"),
        )

    def pod_identity_association(
        self,
        construct_id: str,
        *,
        namespace: str,
        service_account: str,
        role: iam.IRole,
    ) -> eks.CfnPodIdentityAssociation:
        """Bind a pod ServiceAccount to an IAM role via EKS Pod Identity.

        Emits a single ``eks.CfnPodIdentityAssociation`` over a STATIC
        ``pods.eks.amazonaws.com`` trust - the pod->role binding is keyed on
        ``(namespace, serviceAccount)`` and needs NO ServiceAccount
        ``eks.amazonaws.com/role-arn`` annotation on the chart side (proposal
        section 3.5 / 11.2).

        There is NO OIDC provider, NO CfnJson trust map, NO oidc issuer
        derivation, and NO WebIdentityPrincipal - that entire IRSA surface is
        deleted on the Pod Identity path (proposal section 7.4; this is the
        construct's biggest divergence from the VSR ``irsa_role()`` factory). The
        role's trust policy must be a static service-principal grant of
        ``sts:AssumeRole`` + ``sts:TagSession`` to ``pods.eks.amazonaws.com``
        (wired on the role by the composition root, not here).

        The association DependsOn the defensive ``eks-pod-identity-agent``
        add-on so it is guaranteed to resolve regardless of whether Auto Mode
        pre-installs the agent (section 7.4a). Associations are eventually
        consistent, so they are created at provision time in CDK, never in a hot
        startup path (proposal section 3.6).
        """
        assoc = eks.CfnPodIdentityAssociation(
            self,
            construct_id,
            cluster_name=self.cluster_name,
            namespace=namespace,
            service_account=service_account,
            role_arn=role.role_arn,
        )
        # The association must not race the agent add-on.
        assoc.add_dependency(self.pod_identity_addon)
        return assoc

    def enable_container_insights(self, construct_id: str) -> None:
        """Install the amazon-cloudwatch-observability EKS add-on + log group.

        P0-minimal (proposal section 7.5): ships the
        ``amazon-cloudwatch-observability`` add-on (per-pod / per-container
        Container Insights + Fluent Bit container-log forwarding) and a
        CDK-created ``RoutingLogGroup``. The CloudWatch agent gets its
        permissions via EKS Pod Identity (a role assumable by
        ``pods.eks.amazonaws.com`` with ``CloudWatchAgentServerPolicy``,
        associated to the add-on's ServiceAccount).

        NO alarms, NO dashboard, NO SNS topic - those are deferred to P2
        (ADR-0027). This is the major trim from the VSR
        ``enable_container_insights`` (which builds 7 alarms, a dashboard, an SNS
        topic, and the custom Fluent Bit pipeline). The dimensioned
        ``RoutingLatencyByModel`` MetricFilter ships PREP-ONLY / flag-gated off
        (``_ENABLE_ROUTING_LATENCY_BY_MODEL``) because it is data-source-blocked
        until P2; when enabled it dimensions on ``$.["gen_ai.response.model"]``
        (NOT ``$.selected_model``, which RouteIQ never emits - ADR-0027:64-67).
        """
        # Role the CloudWatch agent assumes (Pod Identity). The trust principal
        # is the EKS Pod Identity service principal, with the sts:TagSession +
        # sts:AssumeRole actions Pod Identity requires. ASCII-only description.
        agent_role = iam.Role(
            self,
            f"{construct_id}AgentRole",
            assumed_by=iam.ServicePrincipal("pods.eks.amazonaws.com"),
            description=(
                f"RouteIQ {self.env_name} CloudWatch Observability agent "
                "(Container Insights) Pod Identity"
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy"),
            ],
        )
        assert agent_role.assume_role_policy is not None
        agent_role.assume_role_policy.add_statements(
            iam.PolicyStatement(
                actions=["sts:TagSession"],
                principals=[iam.ServicePrincipal("pods.eks.amazonaws.com")],
            )
        )
        # Bind the agent role to the add-on's ServiceAccount via a Pod Identity
        # association (the add-on's agent SA is cloudwatch-agent in the
        # amazon-cloudwatch namespace). Reuses the helper so it DependsOn the
        # eks-pod-identity-agent add-on.
        assoc = self.pod_identity_association(
            f"{construct_id}AgentPodIdentity",
            namespace="amazon-cloudwatch",
            service_account="cloudwatch-agent",
            role=agent_role,
        )

        addon = eks.CfnAddon(
            self,
            construct_id,
            addon_name="amazon-cloudwatch-observability",
            cluster_name=self.cluster_name,
            resolve_conflicts="OVERWRITE",
        )
        addon.add_dependency(assoc)
        self.container_insights_role = agent_role

        # CDK-created routing log group (proposal section 7.5; vllmsr lesson #8).
        # A CFN AWS::Logs::MetricFilter requires its target log group to ALREADY
        # EXIST, but Fluent Bit's auto_create_group only makes the group at
        # RUNTIME - on a first deploy that ordering fails the whole stack. CDK
        # owning the group breaks the chicken-and-egg: the group exists at deploy
        # time so the metric filters attach, and Fluent Bit's auto_create_group
        # becomes a harmless no-op. RETAIN in non-dev so the captured routing
        # corpus survives a stack teardown; DESTROY in dev for clean re-creates.
        self.routing_log_group_name = f"/aws/containerinsights/{self.cluster_name}/routeiq-routing"
        self.routing_log_group = logs.LogGroup(
            self,
            f"{construct_id}RoutingLogGroup",
            log_group_name=self.routing_log_group_name,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=(
                RemovalPolicy.DESTROY if self.env_name == "dev" else RemovalPolicy.RETAIN
            ),
        )

        # PREP-ONLY / deferred (proposal section 7.5). The dimensioned per-model
        # routing-latency MetricFilter is data-source-blocked until P2: its
        # source (the structured routing_decision CW JSON log line) is a P2
        # BUILD-NEW item, so the filter matches ZERO events at P0. It is
        # flag-gated OFF and authored here so the P2 wave only flips the flag.
        # When enabled it MUST dimension on $.["gen_ai.response.model"] (the
        # telemetry-contract key), NOT $.selected_model (ADR-0027:64-67). AWS
        # forbids default_value on a dimensioned filter, so it is omitted.
        if _ENABLE_ROUTING_LATENCY_BY_MODEL:
            logs.MetricFilter(
                self,
                f"{construct_id}RoutingLatencyByModelFilter",
                log_group=self.routing_log_group,
                filter_pattern=logs.FilterPattern.string_value("$.msg", "=", "routing_decision"),
                metric_namespace=f"routeiq/{self.env_name}/router",
                metric_name="routing_latency_ms_by_model",
                metric_value="$.routing_latency_ms",
                dimensions={"model": _ROUTING_MODEL_DIM_KEY},
            )
