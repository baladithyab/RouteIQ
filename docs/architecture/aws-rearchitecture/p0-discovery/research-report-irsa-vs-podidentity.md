# EKS Auto Mode on CDK: IRSA vs Pod Identity for RouteIQ

For RouteIQ's P0 CDK build on an EKS Auto Mode cluster provisioned through a low-level `CfnCluster`, the pod-IAM decision resolves cleanly: **use EKS Pod Identity, not IAM Roles for Service Accounts (IRSA)**, to grant RouteIQ pods access to Bedrock, S3, and Secrets Manager. AWS recommends Pod Identity specifically for Auto Mode, the Pod Identity Agent is already running on every Auto Mode node, and the CDK association is a four-property resource backed by a static IAM trust policy. The decisive twist is that the L1 `CfnCluster` constraint — which the brief treats as a cost — actually strengthens the case for Pod Identity, because it removes the L2 conveniences that otherwise hide IRSA's considerable complexity.

## 1. Provisioning EKS Auto Mode via CDK: L2 vs L1 CfnCluster

As of mid-2026 there are three CDK paths to an Auto Mode cluster. Being precise about which one the brief has chosen, and what it trades away, matters for the rest of the decision.

The modern L2 path is `aws-cdk-lib/aws-eks-v2`, the now-stable successor to the `@aws-cdk/aws-eks-v2-alpha` module. In this module Auto Mode is the *default* capacity type: the README states that "aws-eks-v2 uses `DefaultCapacityType.AUTOMODE` as the default capacity type" and that "Auto Mode is enabled by default when creating a new cluster without specifying any capacity-related properties" [2]. A bare `new eks.Cluster(this, 'C', { version: KubernetesVersion.V1_34 })` therefore yields an Auto Mode cluster. Critically for a team wary of CDK's historical EKS baggage, aws-eks-v2 is a ground-up rewrite: it uses the "native L1 AWS::EKS::Cluster resource to replace custom resource Custom::AWSCDK-EKS-Cluster," does not create a kubectl handler by default, removes the AwsAuth construct in favor of Access Entry, and lifts the old one-cluster-per-stack and nested-stack limits [2].

That last point reframes the L1 decision. If the reason to reach for L1 `CfnCluster` was to avoid the Lambda-backed custom-resource cluster of the original `aws-eks` module, aws-eks-v2 already delivers exactly that — it emits the same native `AWS::EKS::Cluster` resource you would write by hand, with a more ergonomic surface on top. The honest framing is that L1 buys *maximum* control and a guaranteed zero-Lambda footprint, at the price of writing the access-entry and pod-IAM wiring yourself.

The L1 build itself is straightforward to express. Auto Mode is turned on through three properties on the `CfnCluster`: `computeConfig` (with `enabled`, `nodePools`, and `nodeRoleArn`), `kubernetesNetworkConfig.elasticLoadBalancing`, and `storageConfig.blockStorage` [3].

```ts
const cluster = new eks.CfnCluster(this, 'RouteIqCluster', {
  name: 'routeiq',
  version: '1.34',
  roleArn: clusterRole.roleArn,
  resourcesVpcConfig: { subnetIds, endpointPrivateAccess: true },
  accessConfig: { authenticationMode: 'API', bootstrapClusterCreatorAdminPermissions: true },
  computeConfig: { enabled: true, nodePools: ['system', 'general-purpose'], nodeRoleArn: nodeRole.roleArn },
  kubernetesNetworkConfig: { elasticLoadBalancing: { enabled: true } },
  storageConfig: { blockStorage: { enabled: true } },
});
```

## 2. L1 CfnCluster Gotchas: KubectlProvider, aws-auth, and Access Entries

The L1 path strips every convenience the L2 modules provide, and RouteIQ should budget for the gaps rather than discover them at synth or deploy time.

There is no KubectlProvider. The L2 `Cluster` construct creates a Lambda-backed custom resource that runs `kubectl` during deployment; without it you lose `addManifest`, `addHelmChart`, and — most consequentially for this question — `addServiceAccount`, all of which are kubectl-backed [2]. If RouteIQ needs Kubernetes objects applied at deploy time, it must use a GitOps controller (Argo CD, Flux) or stand up a kubectl provider separately.

There is no aws-auth ConfigMap management and no `grantAccess()` helper. On the L1 path you hand-roll cluster access with the L1 `CfnAccessEntry` resource, which takes a cluster name, principal ARN, type, optional access policies, and Kubernetes groups [4]. EKS Auto Mode softens this considerably: it defaults to `API` authentication mode and auto-creates access entries for its own roles, so "AWS IAM roles are automatically mapped to Kubernetes permissions through EKS access entries, removing the need for manual configuration of aws-auth ConfigMaps" [5]. AWS also advises sticking with the default `API` mode rather than the legacy `CONFIG_MAP_AND_API` for new clusters [1]. But there is a sharp edge: the Auto Mode *node role* needs an access entry of type `EC2`, and access entries of that type cannot carry access policies per an AWS EKS API constraint [4].

```ts
new eks.CfnAccessEntry(this, 'NodeAccess', {
  clusterName: cluster.ref, principalArn: nodeRole.roleArn, type: 'EC2',
});
```

Finally, RouteIQ must create the two Auto Mode IAM roles by hand: a Cluster IAM Role (suggested name `AmazonEKSAutoClusterRole`, with the compute, block-storage, load-balancing, networking, and cluster managed policies) and a Node IAM Role (suggested name `AmazonEKSAutoNodeRole`, with `AmazonEKSWorkerNodeMinimalPolicy` and `AmazonEC2ContainerRegistryPullOnly`) [5].

None of these L1 costs are fixed by the IAM-mechanism choice. They are the price of L1 itself. Choosing Pod Identity removes only the *pod-IAM* slice of L1 friction; the cluster-access and manifest-application slices remain. A claim that "Pod Identity makes L1 easy" overstates it — what is true is that Pod Identity makes the pod-IAM slice nearly free.

## 3. IRSA vs EKS Pod Identity on Auto Mode: Support, Trade-offs, and AWS Direction

Auto Mode supports both IRSA and Pod Identity, but it is built around Pod Identity. The EKS Pod Identity Agent is pre-installed on every Auto Mode cluster: "You do not need to install the EKS Pod Identity Agent on EKS Auto Mode Clusters. This capability is built into EKS Auto Mode" [6]. IRSA is also available — the CDK `IdentityType.IRSA` causes the cluster's OIDC provider to be created when you create a ServiceAccount [7] — but that L2 path depends on the kubectl provider the L1 route omits.

The substantive trade-offs run heavily toward Pod Identity for an in-cloud EKS workload:

| Dimension | IRSA | Pod Identity |
|---|---|---|
| AWS recommendation on Auto Mode | supported | recommended [1] |
| Agent on Auto Mode | n/a | pre-installed [6] |
| Trust policy | per-cluster OIDC, token-keyed condition | static `pods.eks.amazonaws.com` [8] |
| Account scaling | 100 OIDC providers / account | not applicable [8] |
| Role scaling | 2048-byte trust policy (~4–8 trusts) | not applicable [8] |
| Session tags / ABAC | no | yes [8] |
| Cluster readiness | role waits for cluster Ready | role can pre-exist [8] |
| Fargate | yes | no [11] |
| Cross-platform (EKS-A / ROSA / self-managed) | yes | no [8] |
| Cross-account | OIDC-in-account or chained assume | `targetRoleArn` role chaining [10] |
| CDK L1 effort | high (CfnJson + OIDC) | low (4 props) |

IRSA requires a unique IAM OIDC provider per cluster, against a default account limit of 100, and encodes the trust relationship in the role's trust policy, whose 2048-byte size limit caps you at roughly four to eight relationships per role [8]. Pod Identity eliminates both limits: the role trusts the static service principal `pods.eks.amazonaws.com`, can be created before the cluster is ready, is reusable across clusters without trust-policy edits, supports IAM role session tags and ABAC, and exposes a `ListPodIdentityAssociations` inventory API [8]. Pod Identity's own ceiling — up to 5,000 associations per cluster [11] — is far above anything RouteIQ would approach.

Two tensions deserve explicit engagement. First, AWS's "recommended" (Pod Identity) and "still investing in" (IRSA) statements are not a contradiction but a scope statement. AWS says it "plan[s] to continue investing and supporting IRSA," which "works across different Kubernetes deployment options including EKS in the cloud, EKS Anywhere, self-managed Kubernetes clusters on Amazon EC2, and ROSA," while Pod Identity is "purpose built for EKS in the cloud" [8]. RouteIQ's scope — EKS in-cloud, Auto Mode — lands entirely inside the region where AWS names Pod Identity recommended. Second, the general "hand-wire your addons on L1" rule has one important exception that runs in Pod Identity's favor: the one addon Pod Identity depends on, the eks-pod-identity-agent, is exactly the one Auto Mode manages for you [6]. A reader should not count "install the agent" as an L1 cost of choosing Pod Identity — on Auto Mode there is no such step.

## 4. The IRSA CfnJson / KeyMustResolveToString Gotcha and Its Workaround

This is where the L1 constraint becomes decisive. An IRSA IAM role trusts the cluster's OIDC provider through a condition block whose *key* is dynamic:

```jsonc
"Condition": { "StringEquals": {
  "<OIDC_ISSUER>/id/<ID>:sub": "system:serviceaccount:routeiq:routeiq-sa",
  "<OIDC_ISSUER>/id/<ID>:aud": "sts.amazonaws.com"
}}
```

Under CDK, `<OIDC_ISSUER>/id/<ID>` is not known at synth time — it is a token (an unresolved CloudFormation reference). CloudFormation forbids intrinsic functions in dictionary *key* positions because JSON keys must be literal strings. The authoritative CfnJson documentation states the problem verbatim: CfnJson exists "to overcome a limitation in CloudFormation that does not allow using intrinsic functions as dictionary keys (because dictionary keys in JSON must be strings). Specifically this is common in IAM conditions such as `StringEquals: { lhs: \"rhs\" }` where you want lhs to be a reference" [9]. Pass the token straight into a `PolicyStatement` and synth throws — the token cannot resolve to a string for use as a key (the "KeyMustResolveToString"-class failure). A related trap: calling `.replace()` or `.split()` on the raw token to "clean it up" is a silent no-op, because the token is an opaque placeholder, not the resolved string.

The correct workaround wraps the condition object in `CfnJson`, a construct backed by a custom resource that resolves the token at deploy time and exposes the resolved JSON for use in a key position [9].

```ts
const issuer = cluster.attrOpenIdConnectIssuerUrl.replace('https://', '');
const cond = new CfnJson(this, 'IrsaCond', {
  value: {
    [`${issuer}:sub`]: 'system:serviceaccount:routeiq:routeiq-sa',
    [`${issuer}:aud`]: 'sts.amazonaws.com',
  },
});
const role = new iam.Role(this, 'IrsaRole', {
  assumedBy: new iam.OpenIdConnectPrincipal(oidcProvider).withConditions({ StringEquals: cond }),
});
```

On the L2 path `cluster.addServiceAccount()` performs this dance internally — but on the L1 path there is no `addServiceAccount` (no kubectl provider), so RouteIQ would own the CfnJson wrapper, plus the IAM OIDC provider creation, plus the service-account annotation, by hand. Pod Identity needs none of it: because its trust policy is static (`pods.eks.amazonaws.com`), there is no token-keyed condition and therefore no CfnJson. Choosing Pod Identity deletes this entire class of CDK token-resolution failure from the build.

## 5. Creating a Pod Identity Association via CDK and the eks-pod-identity-agent Addon

The Pod Identity build is a single L1 resource per workload. The construct is `CfnPodIdentityAssociation` (CloudFormation `AWS::EKS::PodIdentityAssociation`), with required `clusterName`, `namespace`, `serviceAccount`, and `roleArn`, plus optional `disableSessionTags`, `policy` (a session policy applied as a least-privilege intersection with the role's own policies, so one shared role can serve several service accounts with narrowed effective permissions), `tags`, and `targetRoleArn` for cross-account role chaining [10].

```ts
const podRole = new iam.Role(this, 'RouteIqPodRole', {
  assumedBy: new iam.ServicePrincipal('pods.eks.amazonaws.com'),
});
podRole.assumeRolePolicy?.addStatements(new iam.PolicyStatement({
  actions: ['sts:AssumeRole', 'sts:TagSession'],
  principals: [new iam.ServicePrincipal('pods.eks.amazonaws.com')],
}));
podRole.addToPolicy(new iam.PolicyStatement({
  actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream',
            's3:GetObject', 'secretsmanager:GetSecretValue'],
  resources: [/* scoped ARNs */],
}));
new eks.CfnPodIdentityAssociation(this, 'RouteIqAssoc', {
  clusterName: cluster.ref, namespace: 'routeiq',
  serviceAccount: 'routeiq-sa', roleArn: podRole.roleArn,
});
```

No OIDC provider, no CfnJson, and no addon to install — the eks-pod-identity-agent is pre-installed on Auto Mode [6]. The L2 alternative, `cluster.addServiceAccount('sa', { identityType: eks.IdentityType.POD_IDENTITY })`, auto-installs the agent on a non-Auto-Mode cluster [7], but it requires the kubectl provider that L1 omits, so on the L1 path use the standalone `CfnPodIdentityAssociation`. If RouteIQ's Bedrock, S3, or Secrets Manager resources live in a different account from the cluster, set `targetRoleArn`; Pod Identity then performs role chaining — first assuming the cluster-account role, then the target role — and caches credentials for up to about 59 minutes [10].

## Verdict: IRSA vs Pod Identity for RouteIQ on EKS Auto Mode + CDK L1 CfnCluster

**Use EKS Pod Identity.** The recommendation comes from AWS's own Auto Mode security guidance, not third-party opinion; the agent is already running; the association is four plain properties with a static trust policy; and choosing it deletes the CfnJson / OIDC-provider class of L1 friction that IRSA would otherwise force. The L1 constraint is the clincher rather than a complication: **it strips the L2 `addServiceAccount` helper that normally hides IRSA's complexity**, so on L1 the gap between trivial Pod Identity and hand-rolled IRSA is at its widest.

IRSA would still be the right call under specific conditions: if RouteIQ ran pods on Fargate (Pod Identity is worker-nodes-only [11], though Auto Mode runs on EC2 Managed Instances, so this does not apply), if it had to share IAM trust with non-EKS clusters such as EKS Anywhere or self-managed Kubernetes, if it pinned to an AWS SDK older than the Pod Identity floor, or if it already operated a mature IRSA estate where standardizing on one mechanism would beat the cognitive load of running IRSA and Pod Identity side by side [8]. None of these hold for a greenfield EKS-Auto-Mode-in-cloud workload using current SDKs.

Two honest caveats temper the margin without changing the verdict. First, Pod Identity associations are eventually consistent — AWS warns to "avoid creating or updating associations in critical, high-availability code paths" [11] — so RouteIQ should create them at provision time in CDK, not in a hot startup path; it should keep IPv6 enabled (the agent listens on a link-local IPv6 address) and restrict the IMDS hop limit, which Auto Mode already does via IMDSv2. Second, cross-account is the narrowest gap between the two mechanisms: both work, Pod Identity via a single extra `targetRoleArn` property and IRSA via a provider in the resource account or chained assume. The open question the brief leaves unspecified is RouteIQ's account topology — single-account makes the verdict lopsided, cross-account narrows it but still favors Pod Identity.

A more strategic note, separate from pod IAM: because aws-eks-v2 already emits native L1 CFN with Auto Mode as its default, RouteIQ may want to revisit whether the L1 `CfnCluster` mandate is buying anything the L2 module does not — but that is a question about cluster provisioning, where the answer for pod IAM is Pod Identity either way.

## Sources
[1] EKS Auto Mode Security (best practices). https://docs.aws.amazon.com/eks/latest/best-practices/autosecure.html
[2] AWS CDK aws_eks_v2 README. https://docs.aws.amazon.com/cdk/api/v2/docs/aws-eks-v2-readme.html
[3] AWS::EKS::Cluster ComputeConfig (CloudFormation). https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-eks-cluster-computeconfig.html
[4] class CfnAccessEntry (AWS CDK). https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_eks.CfnAccessEntry.html
[5] Learn about identity and access in EKS Auto Mode. https://docs.aws.amazon.com/eks/latest/userguide/auto-learn-iam.html
[6] Set up the Amazon EKS Pod Identity Agent. https://docs.aws.amazon.com/eks/latest/userguide/pod-id-agent-setup.html
[7] enum IdentityType (aws_eks / aws_eks_v2). https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_eks_v2.IdentityType.html
[8] Amazon EKS Pod Identity: a new way for applications on EKS to obtain IAM credentials. https://aws.amazon.com/blogs/containers/amazon-eks-pod-identity-a-new-way-for-applications-on-eks-to-obtain-iam-credentials/
[9] class CfnJson (AWS CDK). https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.CfnJson.html
[10] AWS::EKS::PodIdentityAssociation (CloudFormation) / CfnPodIdentityAssociation. https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-eks-podidentityassociation.html
[11] Learn how EKS Pod Identity grants pods access to AWS services (limits and restrictions). https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html
