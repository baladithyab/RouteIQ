"""Unit tests for the EFA multi-node NodePool + device-plugin manifests (RouteIQ-2f97).

The C3-deep tier serves a model TOO BIG FOR ONE NODE: one logical replica spans
GPUs across multiple nodes (tensor/pipeline parallel) over AWS EFA RDMA - the only
RDMA fabric on AWS. ``EksClusterConstruct.enable_efa_node_pool`` (C3-deep Shape-1,
Karpenter-native) emits the rendered EFA-capable Karpenter NodePool + EC2 NodeClass
YAML (``EfaNodePoolManifest`` CfnOutput) + the ``eks/aws-efa-k8s-device-plugin``
install hint + the gang-pod EFA contract (``EfaNodePoolManifestDevicePlugin``
CfnOutput) the operator/GitOps applies out-of-band (this construct never uses CDK
``KubernetesManifest`` over the L1 ``CfnCluster``).

The surface is FLAG-GATED off at the composition root (``routeiq:enable_efa`` ->
``RouteIqStack(enable_efa=...)``, DEFAULT OFF), so the default P0 surface
(``make_stack()``) emits ZERO EFA output and the dev snapshot stays byte-stable.
These tests assert the outputs are PRESENT-when-on / ABSENT-when-off and that the
rendered manifest requests ``vpc.amazonaws.com/efa`` on the EFA-capable p5/p6
families.

CRED-FREE / OPERATOR-GATED SPLIT: the manifest authored here is cred-free. The
LIVE half - p5/p6 GPU hardware, the multi-node gang schedule (NVIDIA Grove + KAI
Scheduler, recommended; LWS+Volcano fallback), and the NIXL=LIBFABRIC-not-UCX
runtime probe - is operator-gated.

Synthesised offline via the shared ``make_stack`` / ``template_for`` helpers
(dummy env account ``123456789012`` / ``us-west-2``), credential-free.
"""

from __future__ import annotations

from tests.conftest import make_stack, template_for

_EFA_OUTPUT_KEY = "EfaNodePoolManifest"
_EFA_DEVICE_PLUGIN_KEY = "EfaNodePoolManifestDevicePlugin"


def test_efa_nodepool_absent_by_default() -> None:
    """The default P0 surface (make_stack) emits NO EFA NodePool outputs.

    The composition root calls enable_efa_node_pool only when
    routeiq:enable_efa=True, so the default surface carries zero EFA surface. This
    is the byte-stable guarantee the dev snapshot relies on.
    """
    template = template_for()
    outputs = template.find_outputs("*")
    assert not any(_EFA_OUTPUT_KEY in k for k in outputs), (
        f"EFA NodePool output must be ABSENT on the default surface; got {list(outputs)}"
    )


def test_efa_nodepool_outputs_present_when_enabled() -> None:
    """enable_efa=True emits the EFA NodePool + device-plugin CfnOutputs.

    Exactly two EFA outputs: the NodePool/NodeClass manifest and the device-plugin
    install + gang-pod contract. The device-plugin output key is a superset of the
    NodePool key ("...DevicePlugin"), so count both distinctly.
    """
    template = template_for(enable_efa=True)
    outputs = template.find_outputs("*")
    efa_outputs = [k for k in outputs if _EFA_OUTPUT_KEY in k]
    # NodePool manifest + the device-plugin manifest (its key contains the base key).
    assert any(_EFA_DEVICE_PLUGIN_KEY in k for k in efa_outputs), (
        f"expected an {_EFA_DEVICE_PLUGIN_KEY} output when enabled; got {efa_outputs}"
    )
    nodepool_only = [k for k in efa_outputs if _EFA_DEVICE_PLUGIN_KEY not in k]
    assert len(nodepool_only) == 1, (
        f"expected exactly one {_EFA_OUTPUT_KEY} NodePool output; got {nodepool_only}"
    )


def test_efa_manifest_has_nodepool_and_nodeclass() -> None:
    """The rendered EFA manifest carries both the Karpenter NodePool and the NodeClass."""
    stack = make_stack(enable_efa=True)
    manifest = stack.eks_cluster.efa_node_pool_manifest_value
    assert "kind: NodePool" in manifest
    assert "kind: NodeClass" in manifest
    assert "apiVersion: karpenter.sh/v1" in manifest
    assert "apiVersion: eks.amazonaws.com/v1" in manifest


def test_efa_manifest_selects_efa_capable_p5_p6_families() -> None:
    """The NodePool pins the EFA-capable p5/p6 FAMILIES (not the broad g+p category).

    p5/p5e/p5en + p6/p6e both expose EFA NICs AND carry the local NVMe the KVBM disk
    tier wants; pinning families (not categories) keeps non-EFA GPU nodes off the pool.
    """
    stack = make_stack(enable_efa=True)
    manifest = stack.eks_cluster.efa_node_pool_manifest_value
    assert 'key: "eks.amazonaws.com/instance-family"' in manifest
    for family in ("p5", "p5e", "p5en", "p6", "p6e"):
        assert f'"{family}"' in manifest, f"EFA family {family} missing from manifest"
    # The nvidia.com/gpu NoSchedule taint isolates the (expensive) EFA pool.
    assert "key: nvidia.com/gpu" in manifest
    assert "effect: NoSchedule" in manifest


def test_efa_manifest_is_on_demand_only() -> None:
    """The EFA NodePool is on-demand ONLY (a multi-node gang is all-or-nothing).

    A tensor/pipeline-parallel gang spanning nodes is placement-sensitive +
    all-or-nothing, so spot reclamation mid-gang is fatal. Guard against a
    regression that re-adds spot to the EFA pool.
    """
    stack = make_stack(enable_efa=True)
    manifest = stack.eks_cluster.efa_node_pool_manifest_value
    assert 'key: "karpenter.sh/capacity-type"' in manifest
    assert 'values: ["on-demand"]' in manifest
    assert '"spot"' not in manifest


def test_efa_device_plugin_manifest_requests_efa_resource() -> None:
    """The device-plugin manifest references vpc.amazonaws.com/efa + the gang contract.

    The AWS EFA Kubernetes device plugin advertises vpc.amazonaws.com/efa (what
    NIXL-LIBFABRIC binds to); the gang pod requests it. The seed acceptance: the
    manifest requests vpc.amazonaws.com/efa.
    """
    stack = make_stack(enable_efa=True)
    plugin = stack.eks_cluster.efa_device_plugin_manifest_value
    assert "vpc.amazonaws.com/efa" in plugin
    assert "eks/aws-efa-k8s-device-plugin" in plugin
    # The pod MUST be privileged for NIXL fi_mr_reg on CUDA VRAM (IPC_LOCK alone
    # insufficient) - documented in the contract.
    assert "privileged: true" in plugin


def test_efa_manifest_documents_libfabric_silent_trap() -> None:
    """The manifests document the NIXL=LIBFABRIC-not-UCX silent trap (operator-gated).

    The #1 silent failure: if NIXL lands on UCX it falls back to ~TCP (~100x slower,
    NO error). The probe asserting NIXL=LIBFABRIC is the operator step; CDK cannot
    express it, so it is documented in the rendered manifests.
    """
    stack = make_stack(enable_efa=True)
    plugin = stack.eks_cluster.efa_device_plugin_manifest_value
    assert "LIBFABRIC" in plugin
    assert "UCX" in plugin


def test_efa_nodeclass_reuses_node_role() -> None:
    """The NodeClass carries role: <node-role> - REUSING the Auto Mode node role."""
    stack = make_stack(enable_efa=True)
    manifest = stack.eks_cluster.efa_node_pool_manifest_value
    assert "\n  role: " in manifest
    assert stack.eks_cluster.node_role is not None


def test_efa_manifest_does_not_install_gpu_operator() -> None:
    """Auto Mode supplies the NVIDIA device plugin + drivers - we must NOT install them.

    The EFA device plugin (vpc.amazonaws.com/efa) IS installed out-of-band; the
    NVIDIA GPU Operator is NOT (Auto Mode's accelerated AMI ships the NVIDIA stack).
    """
    stack = make_stack(enable_efa=True)
    manifest = stack.eks_cluster.efa_node_pool_manifest_value
    assert "gpu-operator" not in manifest.lower()


def test_efa_manifest_render_is_deterministic() -> None:
    """Two renders of each EFA manifest are byte-identical (hand-rolled, no PyYAML)."""
    stack = make_stack(enable_efa=True)
    assert stack.eks_cluster.efa_node_pool_manifest() == stack.eks_cluster.efa_node_pool_manifest()
    assert (
        stack.eks_cluster.efa_device_plugin_manifest()
        == stack.eks_cluster.efa_device_plugin_manifest()
    )
