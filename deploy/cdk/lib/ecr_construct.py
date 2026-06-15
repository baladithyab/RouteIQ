"""EcrConstruct - GHCR pull-through cache surface for RouteIQ on AWS.

Ported (parameterized DOWN) from
``/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/ecr_construct.py`` per
``docs/architecture/aws-rearchitecture/31-p0-cdk-foundation-proposal.md`` §9 and
``docs/architecture/aws-rearchitecture/p0-discovery/discover-ecr-ghcr.md``.

RouteIQ publishes its gateway image to ``ghcr.io/baladithyab/routeiq`` today and
keeps doing so UNCHANGED. The pull side is what moves onto AWS: an ECR
pull-through cache (PTC) rule fronts ``ghcr.io`` so private-subnet pods pull the
image through the in-VPC ECR interface endpoints (no GHCR rate limits, scan on
import, immutable-tag governance on the cached copy). The chart's
``image.repository`` is repointed at deploy time to
``<acct>.dkr.ecr.<region>.amazonaws.com/ghcr/baladithyab/routeiq`` (proposal
§9.4 / §11.6) — no Dockerfile, no CI, no registry change.

The construct emits exactly three L1 resources, all governing the PTC-CACHED
``ghcr/*`` repo that ECR auto-creates on first pull (proposal §9.1 — there is NO
standalone ``ecr.Repository``, because the cached repo is the pull target and a
standalone repo would never be written to or read from):

    1. ONE ``ecr.CfnPullThroughCacheRule`` for ``ghcr.io`` (proposal §9.2).
    2. ONE ``ecr.CfnRepositoryCreationTemplate`` (``AppliedFor=PULL_THROUGH_CACHE``,
       ``Prefix=ghcr``, ``ImageTagMutability=IMMUTABLE``) so the auto-created
       ``ghcr/*`` repo is immutable-tagged at CREATION time. ECR's default for an
       un-templated PTC repo is tag-MUTABLE / no-scan, and a PTC-cached repo
       cannot be made immutable retroactively, so the template MUST exist before
       the first pull. ``DependsOn`` the PTC rule (proposal §9.1a / §9.1b).
    3. ONE registry-level ``ecr.CfnRegistryScanningConfiguration`` (``BASIC``,
       ``SCAN_ON_PUSH`` scoped to the ``ghcr/*`` wildcard). Scan-on-push is NOT a
       ``RepositoryCreationTemplate`` property — it is a registry-level setting —
       so this is what makes "scan-on-push on the cached copy" true (proposal
       §9.1a / §9.1c).

Hard constraints honored:

    - **The credential secret is a PARAM, never a literal.** ``credential_arn``
      is passed in by the stack; the construct NEVER creates the secret (it holds
      a real GitHub PAT) and never embeds a literal ARN. The operator provisions
      an ``ecr-pullthroughcache/``-prefixed Secrets Manager secret out-of-band,
      under the AWS-managed ``aws/secretsmanager`` key (ECR rejects a CMK for the
      PTC credential secret), with ``{"username","accessToken"}`` and a
      ``read:packages`` PAT (proposal §9.2; discover-ecr-ghcr §3.2).
    - **Single-account** (operator-confirmed): one deploy account, no
      cross-account ``targetRoleArn`` / ``custom_role_arn`` plumbing.
    - **IMMUTABLE trade-off (stated explicitly):** AWS recommends MUTABLE for PTC
      templates so a re-pushed same-tag upstream image refreshes the cache.
      RouteIQ chooses IMMUTABLE anyway — acceptable because the chart pins by
      ``image.digest`` (takes precedence) and uses immutable release tags, so a
      re-pushed ``1.0.0-rc1`` will not refresh and that is the intended posture
      (proposal §9.1a; chart ``values.yaml:32-39``).
    - **No ECR IAM here.** The CDK/CFN-exec-role and node/pull-identity grants
      (``ecr:CreatePullThroughCacheRule`` / ``ecr:BatchImportUpstreamImage`` etc.)
      live elsewhere and MUST NOT collapse onto the application pod role
      (proposal §9.3 / §4.4).
"""

from __future__ import annotations

from aws_cdk import Aws
from aws_cdk import aws_ecr as ecr
from constructs import Construct

# GHCR PTC upstream contract values (CFN ``AWS::ECR::PullThroughCacheRule``).
# ``UpstreamRegistry`` is an enum; for GHCR it is ``github-container-registry``
# and the URL is ``ghcr.io`` (discover-ecr-ghcr §3.1, CFN TemplateReference).
_GHCR_UPSTREAM_REGISTRY = "github-container-registry"
_GHCR_UPSTREAM_URL = "ghcr.io"

# The destination namespace prefix in the private ECR registry. Cached repos land
# under ``ghcr/*`` (e.g. ``ghcr/baladithyab/routeiq``). This SINGLE value is the
# join point across all three resources: it is the PTC rule's
# ``EcrRepositoryPrefix``, the RepositoryCreationTemplate's ``Prefix`` (which must
# match or the template silently does not apply), and the registry scanning
# filter's ``ghcr/*`` wildcard root.
_GHCR_PREFIX = "ghcr"


class EcrConstruct(Construct):
    """GHCR pull-through cache rule + creation template + registry scan config.

    Emits no standalone ``ecr.Repository`` (proposal §9.1): the pull target is the
    ``ghcr/*`` repo that the PTC rule auto-creates, governed by the
    RepositoryCreationTemplate and the registry scanning configuration below.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        credential_arn: str,
        enable_ghcr_ptc: bool = True,
        **kwargs: object,
    ) -> None:
        """Wire the GHCR PTC surface.

        Args:
            scope: Parent construct (the RouteIqStack).
            construct_id: Logical id for this construct subtree.
            credential_arn: ARN of the operator-provisioned, ``ecr-pullthroughcache/``
                -prefixed Secrets Manager secret holding the GHCR PAT. This is a
                PARAM, NEVER a literal — the construct does NOT create the secret.
            enable_ghcr_ptc: Gate the whole PTC surface (rule + template + scan
                config). When ``False`` the construct emits nothing, and pods pull
                ``ghcr.io`` directly (the chart default) instead of through ECR.
        """
        super().__init__(scope, construct_id, **kwargs)

        self.enable_ghcr_ptc = enable_ghcr_ptc
        self._credential_arn = credential_arn

        self.ptc_rule: ecr.CfnPullThroughCacheRule | None = None
        self.creation_template: ecr.CfnRepositoryCreationTemplate | None = None
        self.scanning_config: ecr.CfnRegistryScanningConfiguration | None = None

        # The chart's ``image.repository`` override value (proposal §7.6 /
        # §9.4 / §11.6). Built from the Aws.ACCOUNT_ID / Aws.REGION CFN tokens so
        # the synth stays account/region-agnostic (no hardcoded account id) and
        # the snapshot baseline does not bake one in. The CfnOutput ``EcrGhcrPrefix``
        # in RouteIqStack surfaces this string; the operator appends the upstream
        # path (``/baladithyab/routeiq``) at ``helm upgrade`` time.
        self.ghcr_prefix_value: str = (
            f"{Aws.ACCOUNT_ID}.dkr.ecr.{Aws.REGION}.amazonaws.com/{_GHCR_PREFIX}"
        )

        if not enable_ghcr_ptc:
            return

        # 1. Pull-through-cache rule (CfnPullThroughCacheRule) for ghcr.io.
        #
        # ``credential_arn`` is supplied by the stack and NEVER a literal: ECR now
        # requires a Secrets Manager credential for the ghcr.io upstream (even for
        # public images) and rejects a credential-less rule at deploy time. The
        # secret itself is operator-provisioned out-of-band (it holds a real
        # GitHub PAT) — this construct only references its ARN.
        self.ptc_rule = ecr.CfnPullThroughCacheRule(
            self,
            "GhcrPullThroughCache",
            ecr_repository_prefix=_GHCR_PREFIX,
            upstream_registry=_GHCR_UPSTREAM_REGISTRY,
            upstream_registry_url=_GHCR_UPSTREAM_URL,
            credential_arn=credential_arn,
        )

        # 2. RepositoryCreationTemplate — immutable-tag governance on the cached repo.
        #
        # ``AppliedFor=["PULL_THROUGH_CACHE"]`` applies the template only to repos
        # ECR auto-creates on first PTC pull. ``Prefix`` MUST equal the PTC rule's
        # ``EcrRepositoryPrefix`` (``ghcr``) or the template silently does not match
        # and the cached repo defaults to MUTABLE/no-scan. The template DependsOn
        # the rule so CFN orders them deterministically and the template exists
        # before any pull (settings apply only at repo-creation time; a PTC-cached
        # repo cannot be made immutable retroactively).
        #
        # NOTE: ``AWS::ECR::RepositoryCreationTemplate`` has NO
        # ``ImageScanningConfiguration`` property — scan-on-push is delivered by the
        # registry scanning config below (step 3), not here (proposal §9.1a).
        self.creation_template = ecr.CfnRepositoryCreationTemplate(
            self,
            "GhcrRepositoryCreationTemplate",
            applied_for=["PULL_THROUGH_CACHE"],
            prefix=_GHCR_PREFIX,
            image_tag_mutability="IMMUTABLE",
            description=(
                "Apply IMMUTABLE tag mutability to repos auto-created by the "
                "ghcr.io pull-through-cache rule. Scan-on-push for these repos "
                "is governed by the registry-level scanning configuration."
            ),
        )
        self.creation_template.add_dependency(self.ptc_rule)

        # 3. Registry-level scan-on-push for the cached ghcr/* repos.
        #
        # Scan-on-push is a REGISTRY setting, not a per-repo / per-template prop, so
        # this is what makes "scan-on-push on the cached copy" true. The filter is
        # scoped to the ``ghcr/*`` wildcard so it does not over-broaden to unrelated
        # repos in the account/region (registry scanning config is account-wide).
        self.scanning_config = ecr.CfnRegistryScanningConfiguration(
            self,
            "RegistryScanningConfiguration",
            scan_type="BASIC",
            rules=[
                ecr.CfnRegistryScanningConfiguration.ScanningRuleProperty(
                    scan_frequency="SCAN_ON_PUSH",
                    repository_filters=[
                        ecr.CfnRegistryScanningConfiguration.RepositoryFilterProperty(
                            filter=f"{_GHCR_PREFIX}/*",
                            filter_type="WILDCARD",
                        ),
                    ],
                ),
            ],
        )
