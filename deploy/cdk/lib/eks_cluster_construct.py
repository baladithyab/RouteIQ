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

from .naming import cluster_name, routing_log_group_name

if TYPE_CHECKING:  # pragma: no cover - forward refs only
    from aws_cdk import aws_ec2 as ec2

# EKS version for the cluster. Auto Mode tracks AMIs itself; this pins the
# control-plane minor (proposal section 7.1).
_EKS_VERSION = "1.33"

# Auto Mode's two built-in node pools. "general-purpose" runs workloads,
# "system" runs cluster-critical add-ons. Both are AWS-managed - RouteIQ does
# NOT author NodePool/EC2NodeClass CRs for them (that is the point of Auto Mode).
_AUTO_MODE_NODE_POOLS = ["general-purpose", "system"]

# -- GPU NodePool + NodeClass (RouteIQ-acdc; flag-gated, DEFAULT OFF) ----------
# The two AWS-managed Auto Mode node pools (general-purpose / system) are
# CPU-ONLY, so a pod requesting ``nvidia.com/gpu`` sits ``Pending`` forever -
# there is no GPU-family NodePool for Karpenter to provision against. To serve
# GPU inference (e.g. a vLLM sidecar / self-hosted model) on this Auto Mode
# cluster the operator must author a CUSTOM Karpenter NodePool selecting GPU
# instance families + a custom EC2 NodeClass that carries the node role.
#
# Auto Mode supplies the NVIDIA device plugin, the GPU drivers, and the
# accelerated Bottlerocket AMI itself when a GPU is requested - so RouteIQ does
# NOT install the NVIDIA device plugin or the GPU Operator (that is the point of
# Auto Mode). These CRs are Kubernetes manifests applied OUT-OF-BAND via
# kubectl/helm/GitOps, exactly like the rest of the app layer (this construct
# never uses CDK ``KubernetesManifest`` over the L1 ``CfnCluster``). The CDK
# deliverable is therefore the RENDERED manifest, surfaced as a ``CfnOutput`` the
# operator/GitOps applies - flag-gated so the default synth/snapshot is byte-stable.
#
# Instance selection: the ``g`` + ``p`` accelerated categories (g6/g6e + the
# p-family) with a generation floor so legacy g4/p3 are excluded; amd64 only
# (Auto Mode is Linux-only and the accelerated AMIs are x86); spot + on-demand so
# Auto Mode can cost-optimize. The ``nvidia.com/gpu`` taint isolates the pool so
# only GPU-tolerating pods land on the (expensive) accelerated nodes.
_GPU_NODE_POOL_NAME = "routeiq-gpu"
_GPU_NODE_CLASS_NAME = "routeiq-gpu"
# Karpenter NodePool requirement keys/values. Authored as a structured dict so
# the test can assert the shape and the YAML render is deterministic.
_GPU_INSTANCE_CATEGORIES = ["g", "p"]
# Gt floor: generation STRICTLY GREATER than this, so "4" -> gen 5+ (g5/g6/g6e +
# p5/p6 family), excluding legacy g4dn/p3/p4 (gen <= 4).
_GPU_INSTANCE_GENERATION_FLOOR = "4"
_GPU_CPU_LIMIT = "1000"
_GPU_MEMORY_LIMIT = "1000Gi"


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
        self.cluster_name = cluster_name(env_name)
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

        NO alarms, NO dashboard, NO SNS topic, and NO routing MetricFilter -
        those are deferred to P2 (ADR-0027). This is the major trim from the VSR
        ``enable_container_insights`` (which builds 7 alarms, a dashboard, an SNS
        topic, and the custom Fluent Bit pipeline). The dimensioned per-model
        routing-latency MetricFilter is OWNED BY P2 now (``ObservabilityConstruct``,
        keyed on ``$.["gen_ai.response.model"]`` over the live ``$.event`` /
        ``$.latency_ms`` contract); P0 only CDK-creates the routing log group the P2
        filters attach to (RouteIQ-8f08 removed the stale P0-side prep filter, which
        keyed on the dead VSR contract no emitter produces).
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
        self.routing_log_group_name = routing_log_group_name(self.env_name)
        self.routing_log_group = logs.LogGroup(
            self,
            f"{construct_id}RoutingLogGroup",
            log_group_name=self.routing_log_group_name,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=(
                RemovalPolicy.DESTROY if self.env_name == "dev" else RemovalPolicy.RETAIN
            ),
        )

        # RouteIQ-8f08: the dimensioned per-model routing-latency MetricFilter is
        # OWNED BY P2 (ObservabilityConstruct), keyed on the live $.event /
        # $.latency_ms contract + $.["gen_ai.response.model"] dimension. The old
        # P0-side prep-only filter keyed on the STALE VSR field contract (which no
        # emitter produces), so even flipped on it matched ZERO events. It is DELETED
        # here (single owner = P2); P0 only owns the log group the P2 filters attach
        # to. P0's default surface emits 0 MetricFilters.

    def gpu_node_pool_manifest(self) -> str:
        """Render the GPU Karpenter NodePool + EC2 NodeClass YAML (RouteIQ-acdc).

        Returns a deterministic, ``kubectl apply -f -`` ready manifest for a
        CUSTOM GPU NodePool + NodeClass on this Auto Mode cluster. It is rendered
        by :meth:`enable_gpu_node_pool` into a ``CfnOutput`` so the operator/GitOps
        applies it out-of-band (this construct never uses CDK
        ``KubernetesManifest`` over the L1 ``CfnCluster`` - the app layer is
        kubectl/helm-applied, section docstring above).

        Shape (re-derived from the AWS EKS Auto Mode GPU-inference guidance):

          * ``NodeClass`` (``eks.amazonaws.com/v1``) carries ``role: <node-role>``
            - REUSING the existing Auto Mode node role (its NAME is the
            :attr:`node_role` ``role_name`` + the ``NodeRoleName`` CfnOutput). The
            existing EC2-type ``CfnAccessEntry`` over that role already permits the
            nodes it launches to join, so no new access entry is needed. Subnet +
            security-group selectors key on the EKS-managed discovery tags Auto
            Mode stamps on the cluster's subnets/SGs.
          * ``NodePool`` (``karpenter.sh/v1``) selects the ``g`` + ``p`` GPU
            instance categories at generation > 4 (gen 5+: g5/g6/g6e + p5/p6,
            excludes legacy g4dn/p3/p4), amd64, spot + on-demand, and carries the
            ``nvidia.com/gpu`` NoSchedule taint so ONLY GPU-tolerating pods land
            on the accelerated nodes.

        Auto Mode supplies the NVIDIA device plugin + drivers + accelerated
        Bottlerocket AMI itself when a pod requests ``nvidia.com/gpu`` - so the
        manifest does NOT install the device plugin or the GPU Operator.

        Hand-rolled (no PyYAML) so the rendered string is byte-deterministic
        across synths (matching the ConfigStateConstruct placeholder approach).
        """
        node_role_name = self.node_role.role_name
        categories = ", ".join(f'"{c}"' for c in _GPU_INSTANCE_CATEGORIES)
        return (
            "# RouteIQ GPU NodePool + EC2 NodeClass (RouteIQ-acdc).\n"
            "# Apply OUT-OF-BAND: kubectl apply -f - <<'EOF' (or via GitOps).\n"
            "# Auto Mode supplies the NVIDIA device plugin + drivers + the\n"
            "# accelerated Bottlerocket AMI; do NOT install the device plugin or\n"
            "# the GPU Operator. GPU pods must tolerate the nvidia.com/gpu taint.\n"
            "---\n"
            "apiVersion: eks.amazonaws.com/v1\n"
            "kind: NodeClass\n"
            "metadata:\n"
            f"  name: {_GPU_NODE_CLASS_NAME}\n"
            "spec:\n"
            # role REUSES the Auto Mode node role (the NodeRoleName CfnOutput).
            f"  role: {node_role_name}\n"
            "  subnetSelectorTerms:\n"
            f"    - tags:\n        kubernetes.io/cluster/{self.cluster_name}: owned\n"
            "  securityGroupSelectorTerms:\n"
            f"    - tags:\n        kubernetes.io/cluster/{self.cluster_name}: owned\n"
            "---\n"
            "apiVersion: karpenter.sh/v1\n"
            "kind: NodePool\n"
            "metadata:\n"
            f"  name: {_GPU_NODE_POOL_NAME}\n"
            "spec:\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            "        routeiq.ai/nodepool: gpu\n"
            "    spec:\n"
            "      nodeClassRef:\n"
            "        group: eks.amazonaws.com\n"
            "        kind: NodeClass\n"
            f"        name: {_GPU_NODE_CLASS_NAME}\n"
            "      taints:\n"
            "        - key: nvidia.com/gpu\n"
            "          effect: NoSchedule\n"
            "      requirements:\n"
            '        - key: "eks.amazonaws.com/instance-category"\n'
            "          operator: In\n"
            f"          values: [{categories}]\n"
            '        - key: "eks.amazonaws.com/instance-generation"\n'
            "          operator: Gt\n"
            f'          values: ["{_GPU_INSTANCE_GENERATION_FLOOR}"]\n'
            '        - key: "kubernetes.io/arch"\n'
            "          operator: In\n"
            '          values: ["amd64"]\n'
            '        - key: "karpenter.sh/capacity-type"\n'
            "          operator: In\n"
            '          values: ["spot", "on-demand"]\n'
            "  limits:\n"
            f'    cpu: "{_GPU_CPU_LIMIT}"\n'
            f"    memory: {_GPU_MEMORY_LIMIT}\n"
        )

    def enable_gpu_node_pool(self, construct_id: str) -> None:
        """Emit the GPU NodePool + NodeClass manifest as an operator CfnOutput.

        Flag-gated by the composition root (``routeiq:enable_gpu_nodepool``,
        DEFAULT OFF). When NOT called the stack carries ZERO GPU surface, so the
        default synth/snapshot is byte-stable; when called it adds a single
        ``CfnOutput`` (``GpuNodePoolManifest``) whose value is the
        :meth:`gpu_node_pool_manifest` YAML the operator/GitOps applies.

        A ``CfnOutput`` (not a ``KubernetesManifest``) is the right surface: this
        construct deliberately applies the entire app/CRD layer out-of-band over
        the L1 ``CfnCluster`` (no L2 ``ICluster`` to ``addManifest`` on), and a
        cred-free ``Template.from_stack`` can assert the output is present-when-on
        / absent-when-off.
        """
        self.gpu_node_pool_manifest_value = self.gpu_node_pool_manifest()
        CfnOutput(
            self,
            construct_id,
            value=self.gpu_node_pool_manifest_value,
            description=(
                "GPU Karpenter NodePool + EC2 NodeClass manifest. Apply "
                "out-of-band (kubectl apply -f - / GitOps) so nvidia.com/gpu pods "
                "schedule on accelerated Auto Mode nodes. Auto Mode supplies the "
                "NVIDIA device plugin + drivers."
            ),
        )
