"""In-process AWS Bedrock model auto-discovery (control-plane).

Enumerates **every** provider's serverless foundation models and their
cross-region inference profiles across a set of source regions, then maps the
result into LiteLLM ``model_list`` entries.  The scan is a *control-plane*
concern -- it uses ``boto3.client("bedrock")`` (NOT ``bedrock-runtime``) and
only ever issues the three read-only control APIs:

* ``ListFoundationModels``    (unfiltered, NOT paginated)
* ``ListInferenceProfiles``   (paginated -- called for BOTH SYSTEM_DEFINED and
                               APPLICATION ``typeEquals`` values)
* ``GetFoundationModelAvailability`` (optional entitlement gate)

Design contracts (RouteIQ-6ae6 / RouteIQ-f86e / RouteIQ-9ea5):

* **Three-way serverless detection.**  A model is serverless when
  ``ON_DEMAND`` *or* ``INFERENCE_PROFILE`` appears in
  ``inferenceTypesSupported``.  We deliberately NEVER pass
  ``byInferenceType=ON_DEMAND`` to ``ListFoundationModels`` -- that filter
  silently drops the INFERENCE_PROFILE-gated frontier models (Claude 4.x,
  Nova, gpt-oss).  ``modelLifecycle.status == ACTIVE`` is the only
  server-side-style filter we apply (client side, post-fetch).
* **Provider-agnostic.**  Records are keyed by ``providerName`` with NO
  hard-coded provider allow-list.  When Bedrock returns a new provider
  (OpenAI ``gpt-oss``, a hypothetical xAI ``grok-*`` if it ever GAs) the scan
  onboards it automatically -- nothing in this module names a specific
  provider.
* **Inference-profile preference hierarchy.**  Per (model, region) the
  selector prefers ``global.*`` > geographic (``us.`` / ``eu.`` / ``apac.`` /
  ``jp.`` / ``au.`` / ``ca.`` / ``us-gov.``) > raw regional modelId, guarded by
  two overrides: (a) skip ``global`` under a residency constraint; (b) only
  pick ``global`` when the model actually exposes a ``global.*`` profile
  (gpt-oss is the counterexample -- it has no global profile, so it falls
  through to the raw modelId on the commercial path and the ``us-gov.`` geo
  profile in GovCloud).

LiteLLM mapping (spec #13-14): system cross-region profile IDs are inlined
into ``litellm_params.model`` as ``bedrock/converse/<tierPrefix><modelId>``
plus ``aws_region_name``; Application Inference Profile ARNs go in a separate
``model_id`` field.  The ``bedrock/`` prefix is always preserved and a cost
entry is registered for ``global.*`` IDs (often missing from LiteLLM's
``model_prices`` JSON, which breaks provider detection -- LiteLLM issue
#17286).

Live multi-region scanning is **operator-gated**: ``BedrockDiscoverySettings``
defaults to ``enabled=False`` so importing or instantiating this module is a
byte-stable no-op until an operator opts in.  All unit tests are credential-free
and mock the boto3 control-plane client with fixtures built from the documented
response shapes -- no live AWS call is ever made offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from litellm_llmrouter.settings import BedrockDiscoverySettings

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

#: Geographic inference-profile prefixes, in no particular order.  ``global``
#: is intentionally NOT in this set -- it is the top tier handled separately.
_GEO_PREFIXES = (
    "us-gov.",  # GovCloud -- MUST be checked before ``us.`` (longer prefix).
    "us.",
    "eu.",
    "apac.",
    "jp.",
    "au.",
    "ca.",
)

#: Lifecycle status that marks a model as generally usable.
_ACTIVE_LIFECYCLE = "ACTIVE"


# ============================================================================
# Tier model
# ============================================================================


class ServerlessClass(str, Enum):
    """How a model is reachable without provisioned throughput.

    ``raw_on_demand``       -- ``ON_DEMAND`` in inferenceTypesSupported; the raw
                               modelId is callable directly in-region.
    ``inference_profile``   -- ONLY ``INFERENCE_PROFILE`` in the supported types;
                               an inference profile is *required* to invoke.
    ``not_serverless``      -- neither token present (PROVISIONED only).
    """

    RAW_ON_DEMAND = "raw_on_demand"
    INFERENCE_PROFILE = "inference_profile"
    NOT_SERVERLESS = "not_serverless"


class ProfileTier(str, Enum):
    """Inference-profile preference tiers (spec #9), highest first."""

    GLOBAL = "global"
    GEOGRAPHIC = "geographic"
    REGIONAL = "regional"


# ============================================================================
# Records
# ============================================================================


@dataclass(frozen=True)
class InferenceProfileRef:
    """A single SYSTEM_DEFINED or APPLICATION inference profile entry.

    ``profile_id`` is the cross-region profile id (e.g.
    ``global.anthropic.claude-...`` or ``us.anthropic.claude-...``) for
    SYSTEM_DEFINED profiles; for APPLICATION profiles the operator-owned ARN is
    carried in ``arn`` and ``profile_id`` mirrors it.
    """

    profile_id: str
    arn: str
    profile_type: str  # "SYSTEM_DEFINED" | "APPLICATION"
    tier: ProfileTier
    geo_prefix: Optional[str] = None  # e.g. "us." for the geographic tier


@dataclass
class DiscoveredModel:
    """A serverless foundation model joined to its inference profiles in a region."""

    model_id: str
    model_arn: str
    provider_name: str
    region: str
    inference_types: tuple[str, ...]
    output_modalities: tuple[str, ...]
    serverless_class: ServerlessClass
    profiles: list[InferenceProfileRef] = field(default_factory=list)
    entitled: Optional[bool] = None  # None == availability not checked.

    @property
    def has_global(self) -> bool:
        return any(p.tier is ProfileTier.GLOBAL for p in self.profiles)

    def geo_profile_for_region(self, region: str) -> Optional[InferenceProfileRef]:
        """Return the in-geo geographic profile for ``region`` if one exists.

        Matches the region's geo family (``us-east-1`` -> ``us.`` /
        ``us-gov.``) against discovered geographic profiles.
        """
        wanted = _region_geo_prefixes(region)
        for pref in wanted:  # ordered most-specific first (us-gov. before us.)
            for p in self.profiles:
                if p.tier is ProfileTier.GEOGRAPHIC and p.geo_prefix == pref:
                    return p
        return None

    def global_profile(self) -> Optional[InferenceProfileRef]:
        for p in self.profiles:
            if p.tier is ProfileTier.GLOBAL:
                return p
        return None


@dataclass(frozen=True)
class ModelSelection:
    """The chosen serverless invocation path for one (model, region)."""

    model_id: str
    region: str
    provider_name: str
    tier: ProfileTier
    serverless_class: ServerlessClass
    #: The id LiteLLM should call: a profile id (system) or the raw modelId.
    invocation_id: str
    #: Application-profile ARN, when the selection is an APPLICATION profile.
    application_profile_arn: Optional[str] = None


class NoServerlessPathInRegion(RuntimeError):
    """Raised when no serverless invocation path exists for a model+region.

    Hit when a model is PROVISIONED-only, or has only a ``global``/foreign-geo
    profile while a residency constraint forbids it and no in-geo geo profile
    or raw ON_DEMAND path exists.
    """


# ============================================================================
# Helpers
# ============================================================================


def _region_geo_prefixes(region: str) -> tuple[str, ...]:
    """Map an AWS region to the geographic profile prefixes that serve it.

    ``us-gov-west-1`` -> ``("us-gov.",)``; ``us-east-1`` -> ``("us.",)``;
    ``eu-west-1`` -> ``("eu.",)``; ``ap-southeast-2`` -> ``("apac.", "au.")``;
    ``ap-northeast-1`` -> ``("apac.", "jp.")``; ``ca-central-1`` -> ``("ca.",)``.

    Ordered most-specific first so a GovCloud region never matches the bare
    ``us.`` commercial geo, and country geos (jp./au.) are preferred over the
    broad ``apac.`` umbrella.
    """
    r = region.lower()
    if r.startswith("us-gov-"):
        return ("us-gov.",)
    if r.startswith("us-"):
        return ("us.",)
    if r.startswith("eu-"):
        return ("eu.",)
    if r.startswith("ca-"):
        return ("ca.",)
    if r.startswith("ap-northeast-"):
        return ("jp.", "apac.")
    if r.startswith("ap-southeast-2") or r.startswith("ap-southeast-4"):
        return ("au.", "apac.")
    if r.startswith("ap-"):
        return ("apac.",)
    return ()


def _classify_profile_tier(profile_id: str) -> tuple[ProfileTier, Optional[str]]:
    """Classify a SYSTEM_DEFINED profile id into its preference tier.

    Returns ``(tier, geo_prefix)`` where ``geo_prefix`` is the matched
    ``us./eu./...`` token for the geographic tier (None otherwise).
    """
    pid = profile_id.lower()
    if pid.startswith("global."):
        return ProfileTier.GLOBAL, None
    for pref in _GEO_PREFIXES:
        if pid.startswith(pref):
            return ProfileTier.GEOGRAPHIC, pref
    return ProfileTier.REGIONAL, None


def classify_serverless(inference_types: Iterable[str]) -> ServerlessClass:
    """THREE-WAY serverless detection (spec #4).

    serverless = ON_DEMAND in types OR INFERENCE_PROFILE in types.
    PROVISIONED-only => NOT serverless.  We never filter byInferenceType.
    """
    types = {str(t).upper() for t in inference_types}
    if "ON_DEMAND" in types:
        return ServerlessClass.RAW_ON_DEMAND
    if "INFERENCE_PROFILE" in types:
        return ServerlessClass.INFERENCE_PROFILE
    return ServerlessClass.NOT_SERVERLESS


# ============================================================================
# Selection function (spec #11-12)
# ============================================================================


def select_invocation_path(
    model: DiscoveredModel,
    *,
    residency_constraint: bool = False,
) -> ModelSelection:
    """Choose the serverless invocation path for one (model, region).

    Preference hierarchy: ``global`` > geographic-in-region > raw modelId
    (only if ``ON_DEMAND``), guarded by:

      (a) skip ``global`` entirely when ``residency_constraint`` is set; and
      (b) only pick ``global`` when the model exposes a ``global.*`` profile.

    Raises :class:`NoServerlessPathInRegion` when none of the tiers yields a
    callable serverless path (PROVISIONED-only, or residency-blocked with no
    in-geo / raw path).
    """
    # (1) global -- only when allowed by residency AND a global profile exists.
    if not residency_constraint:
        gp = model.global_profile()
        if gp is not None:
            return ModelSelection(
                model_id=model.model_id,
                region=model.region,
                provider_name=model.provider_name,
                tier=ProfileTier.GLOBAL,
                serverless_class=model.serverless_class,
                invocation_id=gp.profile_id,
                application_profile_arn=(
                    gp.arn if gp.profile_type == "APPLICATION" else None
                ),
            )

    # (2) geographic profile that serves this model's region.
    geo = model.geo_profile_for_region(model.region)
    if geo is not None:
        return ModelSelection(
            model_id=model.model_id,
            region=model.region,
            provider_name=model.provider_name,
            tier=ProfileTier.GEOGRAPHIC,
            serverless_class=model.serverless_class,
            invocation_id=geo.profile_id,
            application_profile_arn=(
                geo.arn if geo.profile_type == "APPLICATION" else None
            ),
        )

    # (3) raw modelId -- only valid when the model is ON_DEMAND serverless.
    if model.serverless_class is ServerlessClass.RAW_ON_DEMAND:
        return ModelSelection(
            model_id=model.model_id,
            region=model.region,
            provider_name=model.provider_name,
            tier=ProfileTier.REGIONAL,
            serverless_class=ServerlessClass.RAW_ON_DEMAND,
            invocation_id=model.model_id,
        )

    raise NoServerlessPathInRegion(
        f"no serverless invocation path for model={model.model_id!r} in "
        f"region={model.region!r} "
        f"(residency_constraint={residency_constraint}, "
        f"serverless_class={model.serverless_class.value})"
    )


# ============================================================================
# LiteLLM model_list mapping (spec #13-14)
# ============================================================================


def to_litellm_entry(
    selection: ModelSelection,
    *,
    model_name: Optional[str] = None,
    register_cost: bool = True,
) -> dict[str, Any]:
    """Map a :class:`ModelSelection` into a LiteLLM ``model_list`` entry.

    * SYSTEM_DEFINED / regional path: profile id is inlined into
      ``litellm_params.model`` as ``bedrock/converse/<invocationId>`` (the
      ``bedrock/`` prefix is always preserved -- spec #14) and
      ``aws_region_name`` is set.
    * APPLICATION profile: the operator ARN goes into a separate ``model_id``
      field (spec #13).
    * ``register_cost`` adds a ``model_info.input_cost_per_token`` /
      ``output_cost_per_token`` zero-stub for ``global.*`` ids so LiteLLM's
      provider detection does not break on the missing price entry (issue
      #17286). The stub is intentionally 0.0 -- real prices are supplied by the
      operator's price map; the key's mere presence is what unbreaks detection.
    """
    name = (
        model_name
        or f"{selection.tier.value}.{selection.provider_name}.{selection.model_id}"
    )

    litellm_params: dict[str, Any] = {
        "model": f"bedrock/converse/{selection.invocation_id}",
        "aws_region_name": selection.region,
    }
    if selection.application_profile_arn is not None:
        # Application Inference Profile ARNs go in a separate model_id field.
        litellm_params["model_id"] = selection.application_profile_arn

    entry: dict[str, Any] = {
        "model_name": name,
        "litellm_params": litellm_params,
    }

    if register_cost and selection.tier is ProfileTier.GLOBAL:
        # global.* ids are frequently absent from LiteLLM's model_prices JSON,
        # which breaks provider detection (#17286). Registering the keys (even
        # at 0.0) keeps detection working; operators override with real prices.
        entry["model_info"] = {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }

    return entry


# ============================================================================
# Core scan
# ============================================================================


def _list_inference_profiles(client: Any, type_equals: str) -> list[dict[str, Any]]:
    """Page through ``ListInferenceProfiles`` for one ``typeEquals`` value.

    GOTCHA (spec #3): the API defaults to SYSTEM_DEFINED only, so callers MUST
    invoke this for BOTH ``SYSTEM_DEFINED`` and ``APPLICATION``.  Paginated via
    ``nextToken``.
    """
    profiles: list[dict[str, Any]] = []
    next_token: Optional[str] = None
    while True:
        kwargs: dict[str, Any] = {"typeEquals": type_equals}
        if next_token:
            kwargs["nextToken"] = next_token
        resp = client.list_inference_profiles(**kwargs)
        profiles.extend(resp.get("inferenceProfileSummaries", []) or [])
        next_token = resp.get("nextToken")
        if not next_token:
            break
    return profiles


def _index_profiles_by_model_arn(
    profiles: list[dict[str, Any]],
) -> dict[str, list[InferenceProfileRef]]:
    """Index inference-profile summaries by the modelArn they front (spec #3).

    Each profile's ``models[].modelArn`` is the join key back to the foundation
    model.  A profile may front several model ARNs (cross-region families).
    """
    index: dict[str, list[InferenceProfileRef]] = {}
    for p in profiles:
        profile_id = p.get("inferenceProfileId") or ""
        arn = p.get("inferenceProfileArn") or ""
        ptype = p.get("type") or "SYSTEM_DEFINED"
        if ptype == "APPLICATION":
            tier = ProfileTier.REGIONAL
            geo_prefix = None
        else:
            tier, geo_prefix = _classify_profile_tier(profile_id)
        ref = InferenceProfileRef(
            profile_id=profile_id or arn,
            arn=arn,
            profile_type=ptype,
            tier=tier,
            geo_prefix=geo_prefix,
        )
        for m in p.get("models", []) or []:
            model_arn = m.get("modelArn")
            if model_arn:
                index.setdefault(model_arn, []).append(ref)
    return index


def discover_region(
    client: Any,
    region: str,
    *,
    check_availability: bool = False,
) -> list[DiscoveredModel]:
    """Run the discovery ALGORITHM (spec #5) for a single source region.

    Steps:
      1. ``ListFoundationModels`` UNFILTERED (no byInferenceType).
      2. Keep ACTIVE-lifecycle, serverless models (three-way).
      3. ``ListInferenceProfiles`` for BOTH SYSTEM_DEFINED + APPLICATION
         (paginated), indexed by modelArn.
      4. Optional ``GetFoundationModelAvailability`` entitlement gate.
      5. Join into per-(model, region) :class:`DiscoveredModel` records.

    ``client`` is a ``boto3.client("bedrock")`` (control plane). All API calls
    are read-only.
    """
    fm_resp = client.list_foundation_models()
    summaries = fm_resp.get("modelSummaries", []) or []

    profiles = _list_inference_profiles(
        client, "SYSTEM_DEFINED"
    ) + _list_inference_profiles(client, "APPLICATION")
    profile_index = _index_profiles_by_model_arn(profiles)

    discovered: list[DiscoveredModel] = []
    for s in summaries:
        lifecycle = (s.get("modelLifecycle") or {}).get("status")
        if lifecycle != _ACTIVE_LIFECYCLE:
            continue
        inference_types = s.get("inferenceTypesSupported", []) or []
        klass = classify_serverless(inference_types)
        if klass is ServerlessClass.NOT_SERVERLESS:
            continue

        model_id = s.get("modelId") or ""
        model_arn = s.get("modelArn") or ""
        provider = s.get("providerName") or "unknown"
        modalities = tuple(s.get("outputModalities", []) or [])

        entitled: Optional[bool] = None
        if check_availability:
            entitled = _check_availability(client, model_id)
            if entitled is False:
                continue

        discovered.append(
            DiscoveredModel(
                model_id=model_id,
                model_arn=model_arn,
                provider_name=provider,
                region=region,
                inference_types=tuple(inference_types),
                output_modalities=modalities,
                serverless_class=klass,
                profiles=list(profile_index.get(model_arn, [])),
                entitled=entitled,
            )
        )
    return discovered


def _check_availability(client: Any, model_id: str) -> Optional[bool]:
    """Best-effort ``GetFoundationModelAvailability`` entitlement gate.

    Returns ``True``/``False`` for an authoritative answer, or ``None`` when the
    call is unavailable (older boto3, throttled) -- in which case the caller
    treats the model as NOT gated (keeps it).
    """
    fn = getattr(client, "get_foundation_model_availability", None)
    if fn is None:
        return None
    try:
        resp = fn(modelId=model_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("availability check failed for %s: %s", model_id, exc)
        return None
    # GetFoundationModelAvailability interprets four top-level signals:
    #   entitlementAvailability  AVAILABLE | NOT_AVAILABLE  (most direct -- the
    #                            account is entitled to invoke the model)
    #   authorizationStatus      AUTHORIZED | NOT_AUTHORIZED
    #   regionAvailability       AVAILABLE | NOT_AVAILABLE
    #   agreementAvailability.status  AVAILABLE | PENDING | NOT_AVAILABLE | ...
    entitlement = resp.get("entitlementAvailability")
    authorization = resp.get("authorizationStatus")
    region_avail = resp.get("regionAvailability")
    agreement = (resp.get("agreementAvailability") or {}).get("status")
    if entitlement is not None and entitlement != "AVAILABLE":
        return False
    if authorization is not None and authorization != "AUTHORIZED":
        return False
    if region_avail is not None and region_avail != "AVAILABLE":
        return False
    if agreement is not None and agreement not in ("AVAILABLE", "PENDING"):
        return False
    return True


# ============================================================================
# Region resolution (control-plane discipline, mirrors database.py)
# ============================================================================


def _region_from_boto3_session() -> Optional[str]:
    """Last-resort region via boto3's own default-resolution chain.

    Mirrors ``database._region_from_boto3_session`` -- lets boto3 resolve the
    region the way it does for every other AWS call (AWS_REGION /
    AWS_DEFAULT_REGION env, active profile, EC2/ECS IMDS). Wrapped so a
    missing/broken boto3 returns None.
    """
    try:
        import boto3

        return boto3.session.Session().region_name or None
    except Exception:
        return None


def available_bedrock_regions() -> list[str]:
    """Return the regions where the Bedrock *control plane* is offered.

    Uses ``boto3.session.Session().get_available_regions("bedrock")``. Returns
    an empty list when boto3 is absent (the scan then no-ops).
    """
    try:
        import boto3

        return list(boto3.session.Session().get_available_regions("bedrock"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("get_available_regions failed: %s", exc)
        return []


def resolve_source_regions(
    configured: Iterable[str],
    *,
    available: Optional[Iterable[str]] = None,
) -> list[str]:
    """Intersect operator-configured regions with the available Bedrock set.

    When ``configured`` is empty, falls back to the boto3 session default
    region. ``available`` defaults to :func:`available_bedrock_regions`; when it
    is empty (boto3 absent) the configured list passes through unfiltered so a
    fully-mocked test does not need to also mock ``get_available_regions``.
    """
    configured_list = [r.strip() for r in configured if r and r.strip()]
    if not configured_list:
        default = _region_from_boto3_session()
        configured_list = [default] if default else []

    avail = list(available) if available is not None else available_bedrock_regions()
    if not avail:
        return configured_list
    avail_set = set(avail)
    return [r for r in configured_list if r in avail_set]


def _bedrock_control_client(region: str) -> Any:
    """Construct a control-plane ``boto3.client("bedrock")`` for ``region``.

    NOT ``bedrock-runtime`` -- discovery is a control-plane concern. Built
    inline (no module-level cache => no new reset obligation), following the
    ``database._mint_db_token`` pattern.
    """
    import boto3

    return boto3.session.Session().client("bedrock", region_name=region)


# ============================================================================
# Orchestrator
# ============================================================================


@dataclass
class DiscoveryResult:
    """Aggregate result of a multi-region scan."""

    models: list[DiscoveredModel] = field(default_factory=list)
    region_errors: dict[str, str] = field(default_factory=dict)

    def providers(self) -> set[str]:
        return {m.provider_name for m in self.models}

    def to_litellm_model_list(
        self,
        *,
        residency_constraint: bool = False,
        register_cost: bool = True,
    ) -> list[dict[str, Any]]:
        """Map every discovered model to a LiteLLM ``model_list`` entry.

        Models with no serverless path under the given residency constraint are
        skipped (logged at debug) rather than raising -- the orchestrator
        produces a best-effort list.
        """
        entries: list[dict[str, Any]] = []
        for m in self.models:
            try:
                sel = select_invocation_path(
                    m, residency_constraint=residency_constraint
                )
            except NoServerlessPathInRegion as exc:
                logger.debug("skipping %s: %s", m.model_id, exc)
                continue
            entries.append(to_litellm_entry(sel, register_cost=register_cost))
        return entries


def discover_models(
    *,
    settings: Optional["BedrockDiscoverySettings"] = None,
    client_factory: Optional[Any] = None,
) -> DiscoveryResult:
    """Run the multi-region discovery scan (operator-gated).

    ``settings`` defaults to the global ``BedrockDiscoverySettings`` from
    :func:`get_settings`.  Returns an empty :class:`DiscoveryResult` when
    discovery is disabled (the byte-stable default).

    ``client_factory`` is an injectable ``region -> client`` callable used by
    tests to supply a mocked control-plane client; production passes
    :func:`_bedrock_control_client`.  Per-region failures are caught and
    recorded in ``region_errors`` so a single bad region never aborts the scan
    (spec #5: try/except per region).
    """
    if settings is None:
        settings = _get_discovery_settings()
    if not settings.enabled:
        logger.debug("bedrock discovery disabled (enabled=False); no-op")
        return DiscoveryResult()

    regions = resolve_source_regions(settings.source_regions)
    if not regions:
        logger.warning(
            "bedrock discovery enabled but no source regions resolved; "
            "set ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS or AWS_REGION"
        )
        return DiscoveryResult()

    factory = client_factory or _bedrock_control_client
    result = DiscoveryResult()
    for region in regions:
        try:
            client = factory(region)
            models = discover_region(
                client, region, check_availability=settings.check_availability
            )
            result.models.extend(models)
        except Exception as exc:  # try/except per region (spec #5)
            logger.warning("bedrock discovery failed for region %s: %s", region, exc)
            result.region_errors[region] = str(exc)
    return result


# ============================================================================
# Settings glue
# ============================================================================


def _get_discovery_settings() -> "BedrockDiscoverySettings":
    """Lazily fetch the nested discovery settings from the gateway singleton.

    Imported lazily to avoid a settings import side effect at module import.
    """
    from litellm_llmrouter.settings import get_settings

    return get_settings().bedrock_discovery
