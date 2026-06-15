"""Unit tests for the EcrConstruct GHCR pull-through-cache surface (section 12.3).

Asserts the three L1 resources the construct emits to govern the PTC-cached
``ghcr/*`` repo that ECR auto-creates on first pull (proposal section 9), and the
load-bearing NEGATIVE: NO standalone ``AWS::ECR::Repository``.

  1. ONE ``CfnPullThroughCacheRule`` for ``ghcr.io`` (``EcrRepositoryPrefix=ghcr``).
     Its ``CredentialArn`` is a CFN ``Ref`` to the GhcrCredentialSecretArn PARAM,
     NEVER a literal secret (CDK does not create the secret).
  2. ONE ``CfnRepositoryCreationTemplate`` (``AppliedFor=[PULL_THROUGH_CACHE]``,
     ``Prefix=ghcr``, ``ImageTagMutability=IMMUTABLE``) that DependsOn the PTC
     rule so the template exists before the first pull.
  3. ONE registry-level ``CfnRegistryScanningConfiguration`` with
     ``ScanFrequency=SCAN_ON_PUSH`` scoped to the ``ghcr/*`` wildcard (scan-on-push
     is a registry setting, NOT a RepositoryCreationTemplate property).

This replaces the old standalone-repo ``image_tag_mutability`` / scan-on-push prop
assertions: the standalone repo was dropped (proposal section 9.1; it was never
the pull target). Synthesised offline via ``template_for`` (dummy env account
``123456789012`` / ``us-west-2``).
"""

from __future__ import annotations

from tests.conftest import template_for


def test_ghcr_pull_through_cache_rule() -> None:
    """ONE ghcr.io PTC rule with the ghcr prefix; credential is a Ref, not literal."""
    template = template_for()
    rules = template.find_resources("AWS::ECR::PullThroughCacheRule")
    assert len(rules) == 1, f"expected exactly one PTC rule, got {len(rules)}"
    props = next(iter(rules.values()))["Properties"]
    assert props["UpstreamRegistryUrl"] == "ghcr.io", props
    assert props["EcrRepositoryPrefix"] == "ghcr", props
    # The credential MUST be a CFN Ref to the deploy-time param, never a literal
    # ARN string (CDK does not create the GHCR PAT secret; section 9.2).
    cred = props.get("CredentialArn")
    assert isinstance(cred, dict) and "Ref" in cred, (
        "CredentialArn must be a CFN Ref to the GhcrCredentialSecretArn param, "
        f"never a literal secret; got {cred!r}"
    )
    assert not isinstance(cred, str), "CredentialArn must not be a literal string"


def test_repository_creation_template_immutable_pull_through_cache() -> None:
    """ONE RepositoryCreationTemplate: AppliedFor PTC, Prefix ghcr, IMMUTABLE.

    The template carries NO ImageScanningConfiguration property (it is not a
    template prop; scan-on-push is registry-level). DependsOn the PTC rule.
    """
    template = template_for()
    templates = template.find_resources("AWS::ECR::RepositoryCreationTemplate")
    assert len(templates) == 1, f"expected one creation template, got {len(templates)}"
    tpl = next(iter(templates.values()))
    props = tpl["Properties"]
    assert "PULL_THROUGH_CACHE" in props["AppliedFor"], props["AppliedFor"]
    assert props["Prefix"] == "ghcr", props
    assert props["ImageTagMutability"] == "IMMUTABLE", props
    assert "ImageScanningConfiguration" not in props, (
        "RepositoryCreationTemplate has no ImageScanningConfiguration property; "
        "scan-on-push is delivered by the registry scanning configuration"
    )
    # Must DependsOn the PTC rule so it is in place before the first pull
    # auto-creates the cached repo (which cannot be made immutable retroactively).
    depends_on = tpl.get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert depends_on, "creation template must DependsOn the PTC rule"


def test_registry_scan_on_push_ghcr_wildcard() -> None:
    """Registry scanning config: SCAN_ON_PUSH scoped to the ghcr/* wildcard."""
    template = template_for()
    configs = template.find_resources("AWS::ECR::RegistryScanningConfiguration")
    assert len(configs) == 1, f"expected one registry scanning config, got {len(configs)}"
    props = next(iter(configs.values()))["Properties"]
    rules = props["Rules"]
    assert any(r.get("ScanFrequency") == "SCAN_ON_PUSH" for r in rules), rules
    # The filter is scoped to the ghcr/* wildcard so it does not over-broaden.
    filters = [f for r in rules for f in r.get("RepositoryFilters", [])]
    assert any(
        f.get("Filter") == "ghcr/*" and f.get("FilterType") == "WILDCARD" for f in filters
    ), f"expected a ghcr/* WILDCARD scan filter; got {filters}"


def test_no_standalone_ecr_repository() -> None:
    """NO standalone AWS::ECR::Repository (proposal section 9.1).

    The pull target is the ghcr/* repo ECR auto-creates on first pull; a
    standalone ecr.Repository would never be written to or read from, so the
    construct emits none. This is the load-bearing negative.
    """
    template = template_for()
    repos = template.find_resources("AWS::ECR::Repository")
    assert repos == {}, (
        "EcrConstruct must emit NO standalone AWS::ECR::Repository (the pull "
        f"target is the PTC-auto-created ghcr/* repo); found {list(repos)}"
    )


def test_ghcr_ptc_disabled_emits_no_ecr_surface() -> None:
    """enable_ghcr_ptc=False emits NO PTC rule / template / scan config / repo.

    With the flag off, pods pull ghcr.io directly (the chart default) and the
    whole ECR PTC surface is absent (byte-stable default-off path).
    """
    template = template_for(enable_ghcr_ptc=False)
    template.resource_count_is("AWS::ECR::PullThroughCacheRule", 0)
    template.resource_count_is("AWS::ECR::RepositoryCreationTemplate", 0)
    template.resource_count_is("AWS::ECR::RegistryScanningConfiguration", 0)
    template.resource_count_is("AWS::ECR::Repository", 0)
