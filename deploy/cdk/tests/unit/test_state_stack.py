"""P1 state-plane invariants asserted at the INTEGRATED RouteIqStateStack level.

This file is the ground-truth invariant gate the P1 build outline names: it
synthesises the WHOLE ``RouteIqStateStack`` (Aurora + ElastiCache + the
state-stack CMK + the cross-stack pod-role grants, composed exactly as the deploy
wires them) and asserts the load-bearing properties land in the synthesised
template. It is complementary to -- not a duplicate of -- the construct-isolated
tests (``test_replay_store.py`` / ``test_cache.py``, which wrap each construct in
a throwaway stack with a synthetic VPC) and the stack-shape test
(``test_routeiq_state_stack.py``, which proves the cross-stack grant topology /
cycle-avoidance / CfnOutputs). Here the assertions ride the REAL P0-foundation
cross-stack wiring (``foundation=`` by reference, the same path ``app.py`` uses),
so a regression in the stack composition -- not just a construct -- is caught.

Every assertion is offline (``Template.from_stack`` against the dummy env account
``123456789012`` / ``us-west-2``; no AWS creds, no cdk CLI, no network, no
``from_lookup``). The schema-bootstrap Lambda asset is pinned MISSING so the
deterministic inline-fallback path is used (the ``from_asset`` bundling path is
Docker/pip dependent).

Invariants covered (the P1 ground-truth checklist):

  * SEPARATE-STACK invariant: the stores live in ``RouteIqStateStack``, NOT the
    P0 ``RouteIqStack`` (the ADR-0028 ~30-min-rollback blast-radius rule).
  * Aurora ``aurora-postgresql`` with a pinned engine version present (the
    ``cdk-rds-aurora-engine-version-retired-at-deploy`` gotcha: pinned in code as
    a CDK enum, verified live before deploy -- synth cannot catch a retired one).
  * IAM DB authentication enabled.
  * Aurora storage encrypted with a KMS CMK (the state-stack-owned key).
  * 30-day master-secret rotation.
  * The schema-bootstrap CustomResource exists.
  * ElastiCache ``CfnServerlessCache`` engine ``valkey``, MajorEngineVersion "8".
  * The IAM ``CfnUser`` with ``user_id == user_name`` + the ``CfnUserGroup`` +
    the explicit ``add_dependency`` edges (string user-id list needs them).
  * Always-on TLS (no toggle) + KMS at-rest on the cache.
  * Ingress 5432 (Aurora) AND 6379 (cache) from the imported P0 app/pod SG.
  * NO hardcoded secrets and NO hardcoded real account ids in the template
    (credential / secret / account values are CDK tokens or CfnParameters).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from aws_cdk.assertions import Match, Template

from lib import replay_store_construct as rs_module
from tests.conftest import DUMMY_ACCOUNT, make_state_stack


@pytest.fixture(autouse=True)
def _force_inline_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the bootstrap-Lambda asset path MISSING so the state synth is hermetic.

    Present -> ``from_asset`` bundling (Docker/local-pip dependent, NON-hermetic);
    absent -> ``Code.from_inline`` placeholder (deterministic). Forces the
    deterministic path regardless of Docker / whether the asset dir was built.
    """
    monkeypatch.setattr(
        rs_module,
        "_SCHEMA_BOOTSTRAP_ASSET_PATH",
        "/tmp/routeiq-state-stack-invariants-does-not-exist",
        raising=True,
    )


def _state_template(**flags: Any) -> Template:
    _app, _foundation, state = make_state_stack(**flags)
    return Template.from_stack(state)


def _the_one(template: Template, resource_type: str) -> dict:
    resources = template.find_resources(resource_type)
    assert len(resources) == 1, f"expected exactly one {resource_type}, got {len(resources)}"
    return next(iter(resources.values()))


# ----------------------------------------------- separate-stack (rollback) invariant


def test_aurora_lives_in_state_stack_not_p0() -> None:
    """The Aurora cluster is in RouteIqStateStack and ABSENT from the P0 stack.

    This is the ADR-0028 ~30-min-rollback invariant: the state plane is its own
    blast radius so a retired-Aurora-minor / schema-bootstrap regression / cache
    mis-provision rolls back WITHOUT touching the EKS / network / ECR foundation.
    """
    _app, foundation, state = make_state_stack()
    state_t = Template.from_stack(state)
    p0_t = Template.from_stack(foundation)

    state_t.resource_count_is("AWS::RDS::DBCluster", 1)
    p0_t.resource_count_is("AWS::RDS::DBCluster", 0)


def test_cache_lives_in_state_stack_not_p0() -> None:
    """The serverless cache is in RouteIqStateStack and ABSENT from the P0 stack."""
    _app, foundation, state = make_state_stack()
    state_t = Template.from_stack(state)
    p0_t = Template.from_stack(foundation)

    state_t.resource_count_is("AWS::ElastiCache::ServerlessCache", 1)
    p0_t.resource_count_is("AWS::ElastiCache::ServerlessCache", 0)


# ------------------------------------------------------------------ Aurora invariants


def test_aurora_engine_version_present() -> None:
    """The Aurora cluster is aurora-postgresql with a PINNED engine version present.

    The version is pinned in ReplayStoreConstruct as the CDK enum
    ``rds.AuroraPostgresEngineVersion.VER_16_13`` (default; overridable via the
    ``routeiq:state_pg_version`` context key). It renders as the CFN
    ``EngineVersion`` string. NOTE (load-bearing,
    cdk-rds-aurora-engine-version-retired-at-deploy): AWS RETIRES old Aurora
    Postgres minors; this synth/unit assertion proves a version is PINNED, but the
    pinned minor MUST be verified live in us-west-2 with
    ``aws rds describe-db-engine-versions --engine aurora-postgresql --region
    us-west-2`` before deploy -- only a real CreateDBCluster catches a retired one.
    """
    template = _state_template()
    cluster = _the_one(template, "AWS::RDS::DBCluster")["Properties"]
    assert cluster["Engine"] == "aurora-postgresql", cluster
    # A concrete (non-empty) engine version string is pinned, default 16.13.
    assert cluster.get("EngineVersion"), f"Aurora engine version must be pinned; got {cluster!r}"
    assert cluster["EngineVersion"] == "16.13", cluster["EngineVersion"]


def test_aurora_iam_authentication_enabled() -> None:
    """IAM DB authentication is enabled (no static password to the runtime pod)."""
    template = _state_template()
    cluster = _the_one(template, "AWS::RDS::DBCluster")["Properties"]
    assert cluster.get("EnableIAMDatabaseAuthentication") is True, cluster


def test_aurora_storage_encrypted_with_kms_cmk() -> None:
    """Aurora is storage-encrypted and bound to a KMS CMK (the state-stack key)."""
    template = _state_template()
    cluster = _the_one(template, "AWS::RDS::DBCluster")["Properties"]
    assert cluster["StorageEncrypted"] is True, cluster
    # KmsKeyId is a Ref/GetAtt token to the state-stack CMK, never absent/empty.
    assert cluster.get("KmsKeyId"), f"Aurora must reference a KMS CMK; got {cluster!r}"
    # And the state stack owns a CMK with rotation enabled.
    keys = template.find_resources("AWS::KMS::Key")
    rotating = [k for k in keys.values() if k["Properties"].get("EnableKeyRotation") is True]
    assert rotating, "state stack must own a rotation-enabled CMK for the stores"


def test_aurora_secret_rotation_30_days() -> None:
    """A 30-day master-secret rotation schedule + exactly one generated secret."""
    template = _state_template()
    secrets = template.find_resources("AWS::SecretsManager::Secret")
    assert len(secrets) == 1, f"expected one generated master secret, got {len(secrets)}"
    schedules = template.find_resources("AWS::SecretsManager::RotationSchedule")
    assert len(schedules) == 1, f"expected one rotation schedule, got {len(schedules)}"
    rules = next(iter(schedules.values()))["Properties"]["RotationRules"]
    # CDK renders a 30-day cadence as EITHER AutomaticallyAfterDays=30 OR a
    # ScheduleExpression of rate(30 days), depending on the aws-cdk-lib version.
    # Assert the 30-day cadence regardless of which form this version emits.
    assert (
        rules.get("AutomaticallyAfterDays") == 30
        or rules.get("ScheduleExpression") == "rate(30 days)"
    ), f"rotation must be a 30-day cadence; got {rules}"


def test_schema_bootstrap_custom_resource_exists() -> None:
    """The schema-bootstrap CustomResource (Provider-backed) renders in the stack."""
    template = _state_template()
    resources = template.to_json()["Resources"]
    customs = [
        r
        for r in resources.values()
        if r["Type"].startswith("Custom::") or r["Type"] == "AWS::CloudFormation::CustomResource"
    ]
    assert customs, "expected a schema-bootstrap CustomResource in the state stack"
    # The named bootstrap Lambda is also present (the Provider's on_event handler).
    functions = template.find_resources("AWS::Lambda::Function")
    names = [f["Properties"].get("FunctionName") for f in functions.values()]
    assert "routeiq-dev-schema-bootstrap" in names, names


# --------------------------------------------------------------- ElastiCache invariants


def test_cache_engine_valkey_major_8() -> None:
    """The serverless cache is engine valkey with MajorEngineVersion '8' (major-only)."""
    template = _state_template()
    cache = _the_one(template, "AWS::ElastiCache::ServerlessCache")["Properties"]
    assert cache["Engine"] == "valkey", cache
    # MAJOR ONLY -- the split("8.0")[0] derivation, NOT the node-based "8.0".
    assert cache["MajorEngineVersion"] == "8", cache["MajorEngineVersion"]
    assert cache["MajorEngineVersion"] != "8.0"


def test_cache_iam_user_id_equals_user_name() -> None:
    """The IAM-auth CfnUser has user_id == user_name (ElastiCache IAM requirement)."""
    template = _state_template()
    users = template.find_resources("AWS::ElastiCache::User")
    iam_users = [
        u["Properties"]
        for u in users.values()
        if u["Properties"].get("AuthenticationMode") == {"Type": "iam"}
    ]
    assert len(iam_users) == 1, f"expected one IAM-auth user, got {len(iam_users)}"
    props = iam_users[0]
    # IAM auth is rejected by ElastiCache unless UserId == UserName.
    assert props["UserId"] == props["UserName"], props
    assert props["UserId"] == "routeiq-dev-cache-iam-user", props
    assert props["Engine"] == "valkey", props


def test_cache_user_group_and_dependency_edges() -> None:
    """The CfnUserGroup references both users and carries an explicit DependsOn each.

    CDK cannot infer the group->user ordering from a plain string user-id list, so
    both ``add_dependency`` calls MUST render a DependsOn or the group can be
    created before the users exist (deploy race).
    """
    template = _state_template()
    template.resource_count_is("AWS::ElastiCache::User", 2)
    group = _the_one(template, "AWS::ElastiCache::UserGroup")
    props = group["Properties"]
    assert props["UserGroupId"] == "routeiq-dev-cache-ug", props
    assert len(props["UserIds"]) == 2, props["UserIds"]
    depends_on = group.get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert len(depends_on) >= 2, (
        f"user group must DependsOn both CfnUser resources; got {depends_on}"
    )
    # The serverless cache itself DependsOn the user group (deploy ordering).
    cache = _the_one(template, "AWS::ElastiCache::ServerlessCache")
    cache_depends = cache.get("DependsOn") or []
    if isinstance(cache_depends, str):
        cache_depends = [cache_depends]
    assert cache_depends, "serverless cache must DependsOn the user group"


def test_cache_always_on_tls_and_kms() -> None:
    """Always-on TLS (no toggle property) + a KMS CMK at rest on the cache."""
    template = _state_template()
    cache = _the_one(template, "AWS::ElastiCache::ServerlessCache")["Properties"]
    # Serverless TLS is mandatory/always-on: there is NO TransitEncryptionEnabled
    # property to set (the chart pairs this with REDIS_SSL=true on the client).
    assert "TransitEncryptionEnabled" not in cache, (
        f"serverless caches have no TLS toggle (always-on); found {cache!r}"
    )
    # KMS at rest (a Ref/GetAtt token to the state-stack CMK), never absent.
    assert cache.get("KmsKeyId"), f"cache must reference a KMS CMK at rest; got {cache!r}"
    template.has_resource_properties(
        "AWS::ElastiCache::ServerlessCache",
        {"KmsKeyId": Match.any_value()},
    )


# ----------------------------------------------------- ingress from the app/pod SG


def test_aurora_ingress_5432_from_app_sg() -> None:
    """The Aurora SG admits tcp/5432 (from the imported P0 pod SG AND the self-ref)."""
    template = _state_template()
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 5432, "ToPort": 5432, "IpProtocol": "tcp"}},
    )
    # Two rules: the pod-SG peer (the app -> Aurora path) + the SG self-ref (the
    # bootstrap Lambda's ENI shares the SG). Both prove 5432 ingress is wired.
    assert len(ingress) == 2, (
        f"expected two 5432 ingress rules (pod SG + self-ref), got {len(ingress)}"
    )


def test_secrets_endpoint_ingress_443_from_bootstrap_sg() -> None:
    """The P0 vpce_sg admits tcp/443 from the bootstrap Lambda's replay_store_sg.

    Deploy-break guard (RouteIQ-8374): the schema-bootstrap Lambda runs in
    replay_store_sg (private-app tier) and must reach the Secrets Manager interface
    endpoint (private_dns_enabled) on 443 to read the master secret. The P0 vpce_sg
    baseline ingress is 443-from-pod_sg only; this asserts the cross-stack rule that
    ALSO admits replay_store_sg was wired -- without it GetSecretValue hangs and the
    custom resource times out at deploy. The rule is owned by the STATE stack (adding
    it in P0 would close a DependencyCycle), so it renders in the state template.
    """
    template = _state_template()
    ingress_443 = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 443, "ToPort": 443, "IpProtocol": "tcp"}},
    )
    assert len(ingress_443) == 1, (
        f"expected exactly one 443 ingress (bootstrap SG -> Secrets Manager "
        f"endpoint); got {len(ingress_443)}: {list(ingress_443)}"
    )
    props = next(iter(ingress_443.values()))["Properties"]
    assert "CidrIp" not in props, f"443 ingress must be SG-to-SG, not CIDR: {props}"
    assert props.get("SourceSecurityGroupId"), props  # peer = replay_store_sg
    assert props.get("GroupId"), props  # target = imported vpce_sg
    assert props["SourceSecurityGroupId"] != props["GroupId"], props


def test_cache_ingress_6379_from_app_sg() -> None:
    """The cache SG admits tcp/6379 from the imported P0 pod SG (attach_dependencies)."""
    template = _state_template()
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 6379, "ToPort": 6379, "IpProtocol": "tcp"}},
    )
    assert len(ingress) == 1, (
        f"expected one 6379 ingress rule (pod SG -> cache), got {len(ingress)}"
    )


# ------------------------------------------------- no hardcoded secrets / account ids


def test_no_hardcoded_secret_values_in_template() -> None:
    """No literal credential/secret VALUES leak into the synthesised template.

    Aurora's master secret is CDK-generated (the template carries
    ``GenerateSecretString`` directives + token Refs, NOT a plaintext password).
    The ONLY literal allowed is the disabled-default ElastiCache user's inert
    placeholder pw -- it grants nothing (access_string starts with ``off``) and is
    the deploy-required Valkey workaround, not a credential. This scan asserts no
    OTHER plaintext password/secret string appears anywhere in the template.
    """
    from lib import cache_construct as cache_module

    template = _state_template()
    blob = json.dumps(template.to_json())

    # The one tolerated literal (the disabled-default-user workaround pw); assert
    # it is the disabled-user inert placeholder and nothing else carries a literal
    # password. Strip it before the generic scan so the scan stays meaningful.
    placeholder = cache_module._DISABLED_DEFAULT_USER_PASSWORD
    assert placeholder in blob, (
        "the disabled-default-user inert placeholder must be the cache's only "
        "literal pw (it grants nothing: access_string starts with 'off')"
    )
    scrubbed = blob.replace(placeholder, "__DISABLED_DEFAULT_USER_INERT__")

    # No plaintext credential-bearing CFN properties (a generated secret renders
    # GenerateSecretString / SecretStringTemplate, never a resolved Password/
    # MasterUserPassword literal). These keys appearing with a non-token string
    # value would be a leaked credential.
    resources = template.to_json()["Resources"]
    for logical_id, res in resources.items():
        props = res.get("Properties", {})
        for leaky_key in ("MasterUserPassword", "Password"):
            val = props.get(leaky_key)
            if val is None:
                continue
            # A token (dict Ref/GetAtt/Resolve) is fine; a bare str is a leak.
            assert not isinstance(val, str), (
                f"{logical_id}.{leaky_key} must be a token (generated secret), "
                f"not a literal string; got {val!r}"
            )

    # And no obvious "PASSWORD=..."/"SECRET=..." plaintext assignment slipped into
    # an env var / inline body after scrubbing the one tolerated placeholder.
    assert not re.search(r'"(?:DB_PASSWORD|MASTER_PASSWORD)"\s*:\s*"[^"{]', scrubbed), (
        "no plaintext DB_PASSWORD / MASTER_PASSWORD literal may appear in the template"
    )


# AWS-PUBLISHED, AWS-OWNED account ids that legitimately appear as literals in
# the synthesised template -- these are NOT operator/customer account leaks. The
# Secrets Manager RDS rotation Lambda (``add_rotation_single_user``) deploys from
# AWS's public Serverless Application Repository (SAR); CDK bakes the AWS-owned
# SAR publisher account id into the ``serverlessrepo:...:<acct>:applications/...``
# ``applicationId`` ARN, per AWS partition. These are documented AWS constants
# (the same ones every CDK RDS-rotation deploy emits), so they are allow-listed.
_AWS_OWNED_SAR_ROTATION_ACCOUNTS = frozenset(
    {
        "297356227824",  # aws         (us-east-1 SAR publisher)
        "193023089310",  # aws-cn      (cn-north-1 SAR publisher)
        "023102451235",  # aws-us-gov  (us-gov-west-1 SAR publisher)
    }
)


def test_no_hardcoded_real_account_ids_in_template() -> None:
    """No real (non-token, non-dummy, non-AWS-owned) 12-digit account id leaks.

    Cross-stack ARNs are built from ``Aws.ACCOUNT_ID`` (a token resolved at
    deploy), never a literal. The ONLY literal 12-digit ids allowed are:
      * the dummy synth env account ``123456789012`` (the fake test account), and
      * the AWS-OWNED SAR publisher accounts (``_AWS_OWNED_SAR_ROTATION_ACCOUNTS``)
        that CDK bakes into the AWS-managed RDS-rotation ``applicationId`` ARN
        emitted by ``add_rotation_single_user`` -- these are AWS-published
        constants, not operator account ids.
    Any OTHER bare 12-digit id is a hardcoded-account-id leak.
    """
    template = _state_template()
    blob = json.dumps(template.to_json())
    # Find every standalone 12-digit run.
    twelve_digit = re.findall(r"(?<!\d)\d{12}(?!\d)", blob)
    allowed = {DUMMY_ACCOUNT, *_AWS_OWNED_SAR_ROTATION_ACCOUNTS}
    offenders = sorted({d for d in twelve_digit if d not in allowed})
    assert not offenders, (
        f"hardcoded non-dummy, non-AWS-owned 12-digit account id(s) in template: "
        f"{offenders} (use Aws.ACCOUNT_ID tokens, never literals)"
    )

    # Defence in depth: every allow-listed AWS SAR account id, when present, must
    # appear ONLY inside an AWS-managed serverlessrepo applicationId ARN -- never
    # in an operator-facing resource ARN. This stops a real account id from
    # hiding behind the allow-list.
    for sar_acct in _AWS_OWNED_SAR_ROTATION_ACCOUNTS:
        if sar_acct not in blob:
            continue
        for match in re.finditer(rf"arn:[^\"]*?:{sar_acct}:[^\"]*", blob):
            assert ":serverlessrepo:" in match.group(0), (
                f"AWS-owned SAR account {sar_acct} appeared outside a "
                f"serverlessrepo ARN: {match.group(0)!r}"
            )


# --------------------------------------------------------------------- cred-free synth


def test_state_stack_synthesises_offline_against_dummy_env() -> None:
    """The whole composed state stack synthesises with no AWS creds (dummy env)."""
    _app, _foundation, state = make_state_stack()
    assert state.account == DUMMY_ACCOUNT
    # Reaching a rendered template proves the synth needed no creds / no from_lookup.
    assert _the_one(Template.from_stack(state), "AWS::RDS::DBCluster")
