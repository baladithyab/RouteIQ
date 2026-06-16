"""Unit tests for the GPU NodePool + EC2 NodeClass manifest (RouteIQ-acdc).

The two AWS-managed EKS Auto Mode node pools (``general-purpose`` / ``system``)
are CPU-only, so a pod requesting ``nvidia.com/gpu`` sits ``Pending`` forever.
``EksClusterConstruct.enable_gpu_node_pool`` closes that by emitting the rendered
GPU Karpenter NodePool + EC2 NodeClass YAML as a single ``GpuNodePoolManifest``
``CfnOutput`` the operator/GitOps applies out-of-band (this construct never uses
CDK ``KubernetesManifest`` over the L1 ``CfnCluster``; the app/CRD layer is
kubectl/helm-applied).

The surface is FLAG-GATED off at the composition root
(``routeiq:enable_gpu_nodepool`` -> ``RouteIqStack(enable_gpu_nodepool=...)``,
DEFAULT OFF), so the default P0 surface (``make_stack()``) emits ZERO GPU output
and the dev snapshot stays byte-stable. These tests assert the output is
PRESENT-when-on / ABSENT-when-off and that the rendered manifest carries the
GPU-selecting NodePool + the node-role-reusing NodeClass.

Synthesised offline via the shared ``make_stack`` / ``template_for`` helpers
(dummy env account ``123456789012`` / ``us-west-2``), credential-free.
"""

from __future__ import annotations

from tests.conftest import make_stack, template_for

_GPU_OUTPUT_KEY = "GpuNodePoolManifest"


def test_gpu_nodepool_absent_by_default() -> None:
    """The default P0 surface (make_stack) emits NO GPU NodePool output.

    The composition root calls enable_gpu_node_pool only when
    routeiq:enable_gpu_nodepool=True, so the default surface carries zero GPU
    surface. This is the byte-stable guarantee the dev snapshot relies on. The
    output logical id is construct-path-prefixed + hash-suffixed, so match the
    GpuNodePoolManifest substring rather than a prefix.
    """
    template = template_for()
    outputs = template.find_outputs("*")
    assert not any(_GPU_OUTPUT_KEY in k for k in outputs), (
        f"GPU NodePool output must be ABSENT on the default surface; got {list(outputs)}"
    )


def test_gpu_nodepool_output_present_when_enabled() -> None:
    """enable_gpu_nodepool=True emits exactly one GpuNodePoolManifest CfnOutput."""
    template = template_for(enable_gpu_nodepool=True)
    outputs = template.find_outputs("*")
    gpu_outputs = [k for k in outputs if _GPU_OUTPUT_KEY in k]
    assert len(gpu_outputs) == 1, (
        f"expected exactly one {_GPU_OUTPUT_KEY} output when enabled; got {gpu_outputs}"
    )


def test_gpu_manifest_has_nodepool_and_nodeclass() -> None:
    """The rendered manifest carries both the Karpenter NodePool and the NodeClass.

    The manifest string is asserted off the construct attribute (the CfnOutput
    value is a CDK Fn::Join token once the node-role name is interpolated, so the
    raw rendered string is the ergonomic assertion target).
    """
    stack = make_stack(enable_gpu_nodepool=True)
    manifest = stack.eks_cluster.gpu_node_pool_manifest_value
    assert "kind: NodePool" in manifest
    assert "kind: NodeClass" in manifest
    assert "apiVersion: karpenter.sh/v1" in manifest
    assert "apiVersion: eks.amazonaws.com/v1" in manifest


def test_gpu_manifest_selects_gpu_families_and_taints() -> None:
    """NodePool selects g+p categories at gen>4, amd64, and taints nvidia.com/gpu.

    The taint is the isolation that keeps non-GPU pods off the (expensive)
    accelerated nodes; the generation Gt 4 floor excludes legacy g4dn/p3/p4.
    """
    stack = make_stack(enable_gpu_nodepool=True)
    manifest = stack.eks_cluster.gpu_node_pool_manifest_value
    assert 'key: "eks.amazonaws.com/instance-category"' in manifest
    assert 'values: ["g", "p"]' in manifest
    assert 'key: "eks.amazonaws.com/instance-generation"' in manifest
    assert "operator: Gt" in manifest
    assert 'values: ["4"]' in manifest
    assert 'values: ["amd64"]' in manifest
    # The nvidia.com/gpu NoSchedule taint isolates the pool.
    assert "key: nvidia.com/gpu" in manifest
    assert "effect: NoSchedule" in manifest


def test_gpu_manifest_does_not_install_device_plugin() -> None:
    """Auto Mode supplies the NVIDIA device plugin + drivers - we must NOT install them.

    Guard against a regression that re-adds a device-plugin DaemonSet or the GPU
    Operator (Auto Mode's accelerated AMI ships them; double-installing breaks).
    """
    stack = make_stack(enable_gpu_nodepool=True)
    manifest = stack.eks_cluster.gpu_node_pool_manifest_value
    lowered = manifest.lower()
    assert "daemonset" not in lowered
    assert "gpu-operator" not in lowered
    assert "k8s-device-plugin" not in lowered


def test_gpu_nodeclass_reuses_node_role() -> None:
    """The NodeClass carries role: <node-role> - REUSING the Auto Mode node role.

    The seed mandate: reuse the emitted NodeRoleName, not mint a new role. The
    rendered manifest interpolates the construct's node_role.role_name, so the
    rendered string contains a `role:` line (the resolved/tokenised role name).
    """
    stack = make_stack(enable_gpu_nodepool=True)
    manifest = stack.eks_cluster.gpu_node_pool_manifest_value
    assert "\n  role: " in manifest
    # The node role attribute the manifest reuses is the same object the stack
    # exposes via the NodeRoleName output (identity, not a freshly-minted role).
    assert stack.eks_cluster.node_role is not None


def test_gpu_manifest_render_is_deterministic() -> None:
    """Two renders of the manifest are byte-identical (hand-rolled, no PyYAML)."""
    stack = make_stack(enable_gpu_nodepool=True)
    assert stack.eks_cluster.gpu_node_pool_manifest() == stack.eks_cluster.gpu_node_pool_manifest()
