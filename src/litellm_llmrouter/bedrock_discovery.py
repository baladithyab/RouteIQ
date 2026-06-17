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


@dataclass
class MarketplaceEndpoint:
    """An Amazon Bedrock Marketplace / mantle custom-deployment endpoint.

    Surfaced by ``ListMarketplaceModelEndpoints`` (RouteIQ-7105). ``endpoint_arn``
    is the operator-owned, invocable endpoint ARN (``bedrock/<endpointArn>``);
    ``model_source`` is the Marketplace model ARN it deploys. Only ``REGISTERED``
    endpoints are routable -- an ``INCOMPATIBLE_ENDPOINT`` is skipped.
    """

    endpoint_arn: str
    model_source: str
    region: str
    status: str = "REGISTERED"

    @property
    def is_routable(self) -> bool:
        return (self.status or "").upper() == "REGISTERED"


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


def _marketplace_model_name(endpoint: MarketplaceEndpoint) -> str:
    """Derive a stable ``model_name`` for a marketplace endpoint arm.

    Uses the last ARN segment (the endpoint name) prefixed with ``marketplace.``
    so the synthesized name is human-recognisable and collision-resistant. Falls
    back to the full ARN when it has no ``/`` segment.
    """
    tail = endpoint.endpoint_arn.rsplit("/", 1)[-1] or endpoint.endpoint_arn
    return f"marketplace.{tail}"


def marketplace_to_litellm_entry(
    endpoint: MarketplaceEndpoint,
    *,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Map a :class:`MarketplaceEndpoint` into a LiteLLM ``model_list`` entry.

    The endpoint ARN is invocable directly -- it is inlined into
    ``litellm_params.model`` as ``bedrock/<endpointArn>`` (the ``bedrock/``
    prefix is preserved, mirroring :func:`to_litellm_entry`) with
    ``aws_region_name`` set. ``model_info`` records the marketplace model source
    and an ``arm_id`` so telemetry can attribute the arm.
    """
    name = model_name or _marketplace_model_name(endpoint)
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"bedrock/{endpoint.endpoint_arn}",
            "aws_region_name": endpoint.region,
        },
        "model_info": {
            "arm_id": f"{endpoint.region}/{endpoint.endpoint_arn}",
            "marketplace_model_source": endpoint.model_source,
            "tier": "marketplace",
        },
    }


def _list_marketplace_endpoints(client: Any) -> list[dict[str, Any]]:
    """Page through ``ListMarketplaceModelEndpoints`` (RouteIQ-7105).

    Returns the raw ``marketplaceModelEndpoints`` summaries. Defensive: when the
    boto3 client predates the API (no ``list_marketplace_model_endpoints``
    attribute) returns an empty list so the FM scan is unaffected.
    """
    fn = getattr(client, "list_marketplace_model_endpoints", None)
    if fn is None:
        return []
    endpoints: list[dict[str, Any]] = []
    next_token: Optional[str] = None
    while True:
        kwargs: dict[str, Any] = {}
        if next_token:
            kwargs["nextToken"] = next_token
        resp = fn(**kwargs)
        endpoints.extend(resp.get("marketplaceModelEndpoints", []) or [])
        next_token = resp.get("nextToken")
        if not next_token:
            break
    return endpoints


def discover_marketplace_endpoints(
    client: Any, region: str
) -> list[MarketplaceEndpoint]:
    """Enumerate Marketplace / mantle custom-deployment endpoints in a region.

    Joins ``ListMarketplaceModelEndpoints`` summaries into
    :class:`MarketplaceEndpoint` records, dropping any with an empty
    ``endpointArn``. Non-``REGISTERED`` endpoints are kept here (the routable
    filter is applied at mapping time) so callers can inspect drift/status.
    """
    out: list[MarketplaceEndpoint] = []
    for s in _list_marketplace_endpoints(client):
        arn = s.get("endpointArn") or ""
        if not arn:
            continue
        out.append(
            MarketplaceEndpoint(
                endpoint_arn=arn,
                model_source=s.get("modelSourceIdentifier") or "",
                region=region,
                status=s.get("status") or "REGISTERED",
            )
        )
    return out


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

    Marketplace / mantle custom-deployment endpoints (RouteIQ-7105) are scanned
    separately via :func:`discover_marketplace_endpoints` (called by the
    orchestrator under its own flag), not here, so the serverless-FM path stays
    byte-stable.
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
    #: Marketplace / mantle custom-deployment endpoints (RouteIQ-7105).
    marketplace_endpoints: list[MarketplaceEndpoint] = field(default_factory=list)

    def providers(self) -> set[str]:
        return {m.provider_name for m in self.models}

    def _marketplace_entries(
        self, *, group_name: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Map every ROUTABLE marketplace endpoint to a model_list entry.

        ``group_name`` (when set, the auto-group case) collapses each endpoint
        arm under the shared group name; otherwise each endpoint keeps its own
        ``marketplace.<name>`` model_name. Non-``REGISTERED`` endpoints are
        skipped. Dedups identical (region, endpoint) arms discovered twice.
        """
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ep in self.marketplace_endpoints:
            if not ep.is_routable:
                logger.debug(
                    "skipping non-routable marketplace endpoint %s", ep.endpoint_arn
                )
                continue
            arm_id = f"{ep.region}/{ep.endpoint_arn}"
            if arm_id in seen:
                continue
            seen.add(arm_id)
            entries.append(marketplace_to_litellm_entry(ep, model_name=group_name))
        return entries

    def to_litellm_model_list(
        self,
        *,
        residency_constraint: bool = False,
        register_cost: bool = True,
        auto_group: bool = False,
        auto_group_name: str = "claude-auto",
        synthesis_mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Map every discovered model to a LiteLLM ``model_list`` entry.

        Models with no serverless path under the given residency constraint are
        skipped (logged at debug) rather than raising -- the orchestrator
        produces a best-effort list.

        ``synthesis_mode`` (RouteIQ-77e8) selects HOW the discovered arms fold
        into the list and, when given, SUPERSEDES the legacy ``auto_group`` bool:

        * ``"distinct"`` (default when neither arg is set) -- one distinct
          ``model_name`` per discovered arm (byte-stable historical behavior).
        * ``"auto_group"`` -- every arm collapses onto the single
          ``auto_group_name`` group (delegates to
          :meth:`to_auto_group_model_list`).
        * ``"logical_groups"`` -- one group PER LOGICAL model fanned across its
          global/geo/regional/mantle arms (delegates to
          :meth:`synthesize_model_groups`). Closes RouteIQ-1c9d's built-not-wired
          gap.

        For back-compat, when ``synthesis_mode`` is None the legacy
        ``auto_group=True`` maps to ``"auto_group"`` (and ``auto_group=False`` to
        ``"distinct"``). LiteLLM treats every ``model_list`` row sharing a
        ``model_name`` as an arm of that group, which the routing strategy (e.g.
        the Kumaraswamy-Thompson bandit) then picks between.
        """
        mode = synthesis_mode or ("auto_group" if auto_group else "distinct")
        if mode == "auto_group":
            return self.to_auto_group_model_list(
                group_name=auto_group_name,
                residency_constraint=residency_constraint,
                register_cost=register_cost,
            )
        if mode == "logical_groups":
            return self.synthesize_model_groups(
                residency_constraint=residency_constraint,
                register_cost=register_cost,
            )
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
        # Marketplace / mantle endpoints (RouteIQ-7105): each keeps its own
        # distinct ``marketplace.<name>`` model_name in the non-auto-group case.
        entries.extend(self._marketplace_entries())
        return entries

    def to_auto_group_model_list(
        self,
        *,
        group_name: str = "claude-auto",
        residency_constraint: bool = False,
        register_cost: bool = True,
    ) -> list[dict[str, Any]]:
        """Synthesize a SINGLE mixed-Bedrock routing group from discovery.

        Every discovered serverless model that yields a callable invocation path
        becomes one arm of a group all sharing ``model_name == group_name``
        (default ``"claude-auto"``). The arms span every tier and provider the
        scan found -- Claude, Nova, gpt-oss, and any future provider -- so a
        routing strategy can pick the cheapest/best arm for each request.

        This is the recipe behind the Claude-Code cost-routing use case: point
        an Anthropic client's ``model`` at ``claude-auto`` and let the
        Kumaraswamy-Thompson bandit (or any ``routing_strategy``) cascade across
        the mixed Bedrock arms instead of pinning Opus.

        Each arm carries a ``model_info`` ``arm_id`` (its distinct upstream id)
        so operators and telemetry can still see which physical model an arm
        maps to -- LiteLLM only requires ``model_name`` to be shared, not unique.

        Models with no serverless path under ``residency_constraint`` are skipped
        (logged at debug). Returns an empty list when no model yields a path
        (e.g. discovery disabled / no models found) -- the caller then leaves the
        operator-authored ``model_list`` untouched.
        """
        entries: list[dict[str, Any]] = []
        seen_arms: set[str] = set()
        for m in self.models:
            try:
                sel = select_invocation_path(
                    m, residency_constraint=residency_constraint
                )
            except NoServerlessPathInRegion as exc:
                logger.debug("skipping %s for auto-group: %s", m.model_id, exc)
                continue
            entry = to_litellm_entry(
                sel, model_name=group_name, register_cost=register_cost
            )
            # arm_id makes each arm distinguishable in telemetry/inspection even
            # though all arms share the group's model_name. Dedup identical arms
            # discovered from multiple source regions (same invocation path).
            arm_id = f"{sel.region}/{sel.invocation_id}"
            if arm_id in seen_arms:
                continue
            seen_arms.add(arm_id)
            info = entry.setdefault("model_info", {})
            info["arm_id"] = arm_id
            info["provider"] = sel.provider_name
            info["tier"] = sel.tier.value
            entries.append(entry)
        # Marketplace / mantle endpoints (RouteIQ-7105) join the SAME group as
        # additional arms when auto-grouping, so the bandit can cascade across
        # FM + custom-deployment arms alike. Dedup is internal to the helper.
        for mp_entry in self._marketplace_entries(group_name=group_name):
            arm_id = mp_entry["model_info"]["arm_id"]
            if arm_id in seen_arms:
                continue
            seen_arms.add(arm_id)
            entries.append(mp_entry)
        return entries

    def synthesize_model_groups(
        self,
        *,
        residency_constraint: bool = False,
        register_cost: bool = True,
        name_prefix: str = "",
        bind_marketplace: bool = True,
    ) -> list[dict[str, Any]]:
        """Bind each LOGICAL model into ONE group fanned out over its arms
        (RouteIQ-1c9d).

        Unlike :meth:`to_auto_group_model_list` (which collapses EVERY discovered
        model onto a single shared group), this binds arms by their *logical
        model identity*: ``global.anthropic.claude-sonnet-4-v1:0`` discovered in
        ``us-east-1``, the ``us.`` geo profile in ``us-west-2``, and a raw
        ``anthropic.claude-sonnet-4-v1:0`` ON_DEMAND arm in ``eu-west-1`` all land
        under the SINGLE ``model_name`` ``<name_prefix>anthropic.claude-sonnet-4``
        as three arms. So one logical ``model_name`` fans out across region /
        account / mantle arms automatically and a routing strategy picks the best
        arm per request.

        Each arm carries a ``model_info`` ``arm_id`` (its distinct
        ``region/invocation_id``), ``provider``, ``tier``, and ``region`` so
        telemetry can still attribute the physical arm. Arms with an identical
        ``arm_id`` discovered from multiple source regions are deduped.

        ``bind_marketplace`` (default True) also binds each routable marketplace /
        mantle endpoint into the logical group derived from its model source, so a
        custom-deployment arm of the same model joins its serverless siblings.
        Models with no serverless path under ``residency_constraint`` are skipped
        (logged at debug). Returns an empty list when nothing was discovered.
        """
        # logical model_name -> list of arm entries.
        groups: dict[str, list[dict[str, Any]]] = {}
        seen_arms: set[str] = set()

        def _add_arm(group: str, entry: dict[str, Any], arm_id: str) -> None:
            if arm_id in seen_arms:
                return
            seen_arms.add(arm_id)
            info = entry.setdefault("model_info", {})
            info["arm_id"] = arm_id
            groups.setdefault(group, []).append(entry)

        for m in self.models:
            try:
                sel = select_invocation_path(
                    m, residency_constraint=residency_constraint
                )
            except NoServerlessPathInRegion as exc:
                logger.debug("skipping %s for model-group bind: %s", m.model_id, exc)
                continue
            logical = f"{name_prefix}{_logical_model_name(m.model_id, m.provider_name)}"
            entry = to_litellm_entry(
                sel, model_name=logical, register_cost=register_cost
            )
            arm_id = f"{sel.region}/{sel.invocation_id}"
            info = entry.setdefault("model_info", {})
            info["provider"] = sel.provider_name
            info["tier"] = sel.tier.value
            info["region"] = sel.region
            _add_arm(logical, entry, arm_id)

        if bind_marketplace:
            for ep in self.marketplace_endpoints:
                if not ep.is_routable:
                    continue
                # The model source is a marketplace-model ARN; take the tail (the
                # model id) so a mantle custom-deployment of the SAME logical model
                # binds into that model's group, not a separate ARN-named group.
                source_id = (ep.model_source or "").rsplit("/", 1)[-1]
                logical = (
                    f"{name_prefix}{_logical_model_name(source_id, 'marketplace')}"
                )
                entry = marketplace_to_litellm_entry(ep, model_name=logical)
                arm_id = entry["model_info"]["arm_id"]
                _add_arm(logical, entry, arm_id)

        # Flatten in deterministic group order; arms keep discovery order.
        out: list[dict[str, Any]] = []
        for group in sorted(groups):
            out.extend(groups[group])
        return out


def _logical_model_name(model_id: str, provider_name: str) -> str:
    """Derive a stable LOGICAL model name from a Bedrock modelId (RouteIQ-1c9d).

    The same logical model is offered across many regions/accounts under distinct
    invocation ids (raw modelId, ``us.``/``eu.`` geo profiles, ``global.``
    profiles). To bind those arms under ONE ``model_name`` we strip the
    region-varying parts:

      * any leading tier/geo prefix (``global.`` / ``us.`` / ``eu.`` / ...), and
      * the trailing Bedrock version suffix (``-v1:0`` / ``:0``).

    so ``global.anthropic.claude-sonnet-4-v1:0``, ``us.anthropic.claude-sonnet-4``
    and the raw ``anthropic.claude-sonnet-4-v1:0`` all collapse to one logical
    ``anthropic.claude-sonnet-4`` group. The provider name is only used as a
    fallback when the modelId is empty.
    """
    base = model_id or provider_name.lower()
    lowered = base.lower()
    # Strip a leading tier prefix (global.) or geo prefix (us./eu./...).
    if lowered.startswith("global."):
        base = base[len("global.") :]
    else:
        for pref in _GEO_PREFIXES:
            if lowered.startswith(pref):
                base = base[len(pref) :]
                break
    # Strip the trailing Bedrock version suffix: "-v<NN>:<MM>" or ":<MM>".
    import re

    base = re.sub(r"-v\d+:\d+$", "", base)
    base = re.sub(r":\d+$", "", base)
    return base or (model_id or provider_name.lower())


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
    # Resolve through the effective-setting helpers so the ``full_bedrock_coverage``
    # rollup (RouteIQ-77e8) turns discovery + marketplace on with one flag. Fall
    # back to the raw fields for any settings shim lacking the helpers.
    effective_enabled = getattr(settings, "effective_enabled", settings.enabled)
    if not effective_enabled:
        logger.debug("bedrock discovery disabled (effective_enabled=False); no-op")
        return DiscoveryResult()

    regions = resolve_source_regions(settings.source_regions)
    if not regions:
        logger.warning(
            "bedrock discovery enabled but no source regions resolved; "
            "set ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS or AWS_REGION"
        )
        return DiscoveryResult()

    include_marketplace = getattr(
        settings,
        "effective_include_marketplace_endpoints",
        getattr(settings, "include_marketplace_endpoints", False),
    )
    factory = client_factory or _bedrock_control_client
    result = DiscoveryResult()
    for region in regions:
        try:
            client = factory(region)
            models = discover_region(
                client, region, check_availability=settings.check_availability
            )
            result.models.extend(models)
            # Marketplace / mantle custom-deployment endpoints (RouteIQ-7105),
            # opt-in. Failure here must not drop the FM models already collected
            # for this region, so it is guarded independently.
            if include_marketplace:
                try:
                    result.marketplace_endpoints.extend(
                        discover_marketplace_endpoints(client, region)
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "marketplace endpoint scan failed for region %s: %s",
                        region,
                        exc,
                    )
        except Exception as exc:  # try/except per region (spec #5)
            logger.warning("bedrock discovery failed for region %s: %s", region, exc)
            result.region_errors[region] = str(exc)
    return result


# ============================================================================
# Catalogue drift detection (RouteIQ-9a42)
# ============================================================================


@dataclass(frozen=True)
class CatalogueDrift:
    """Diff of the DISCOVERED catalogue (models_available) vs CONFIGURED.

    ``available`` is the set of invocation ids the scan found Bedrock offers;
    ``configured`` is the set of bedrock invocation ids in the live model_list.
    ``missing`` = available − configured (offered but NOT routed: under-served);
    ``extra`` = configured − available (routed but NOT offered: stale/dangling).
    """

    available: frozenset[str]
    configured: frozenset[str]

    @property
    def missing(self) -> frozenset[str]:
        return self.available - self.configured

    @property
    def extra(self) -> frozenset[str]:
        return self.configured - self.available

    @property
    def drift_count(self) -> int:
        return len(self.missing) + len(self.extra)

    @property
    def has_drift(self) -> bool:
        return self.drift_count > 0


def _invocation_id_from_litellm_model(model: str) -> Optional[str]:
    """Extract the bedrock invocation id from a ``litellm_params.model`` string.

    Strips the ``bedrock/`` provider prefix and an optional ``converse/`` route
    so the id lines up with what the discovery selection produces. Returns None
    for a non-bedrock model string (it is not part of the Bedrock catalogue).
    """
    if not model or not model.startswith("bedrock/"):
        return None
    rest = model[len("bedrock/") :]
    if rest.startswith("converse/"):
        rest = rest[len("converse/") :]
    return rest or None


def compute_catalogue_drift(
    result: "DiscoveryResult",
    configured_model_list: Iterable[dict[str, Any]],
    *,
    residency_constraint: bool = False,
) -> CatalogueDrift:
    """Compute model-catalogue drift (RouteIQ-9a42).

    ``models_available`` is derived from the discovery ``result`` (the serverless
    invocation ids the scan selected, plus routable marketplace endpoint ARNs);
    ``models_configured`` is the set of bedrock invocation ids parsed out of the
    live ``configured_model_list`` (non-bedrock arms are ignored -- they are not
    part of the Bedrock catalogue).
    """
    available: set[str] = set()
    for m in result.models:
        try:
            sel = select_invocation_path(m, residency_constraint=residency_constraint)
        except NoServerlessPathInRegion:
            continue
        available.add(sel.invocation_id)
    for ep in result.marketplace_endpoints:
        if ep.is_routable:
            available.add(ep.endpoint_arn)

    configured: set[str] = set()
    for entry in configured_model_list or []:
        params = entry.get("litellm_params") or {}
        inv = _invocation_id_from_litellm_model(params.get("model", ""))
        if inv is not None:
            configured.add(inv)

    return CatalogueDrift(
        available=frozenset(available), configured=frozenset(configured)
    )


#: Lazily-created OTel counter for catalogue-drift signals (RouteIQ-9a42). Kept
#: module-local so this control-plane module owns its own instrument without a
#: cross-cluster edit to ``metrics.py``. Reset via :func:`reset_drift_metric`.
_drift_counter: Any = None


def _get_drift_counter() -> Any:
    """Return (lazily creating) the catalogue-drift OTel counter, or None.

    Uses the shared OTel meter from :mod:`observability`. Returns None when OTel
    is unavailable so :func:`record_catalogue_drift_metric` is a clean no-op.
    """
    global _drift_counter
    if _drift_counter is not None:
        return _drift_counter
    try:
        from litellm_llmrouter.observability import get_meter

        meter = get_meter()
        if meter is None:
            return None
        _drift_counter = meter.create_counter(
            name="gateway.bedrock.catalogue_drift",
            description=(
                "Bedrock model-catalogue drift: discovered models_available vs "
                "model_list models_configured, labelled by direction "
                "(missing=offered-not-routed / extra=routed-not-offered)"
            ),
            unit="{model}",
        )
        return _drift_counter
    except Exception:  # pragma: no cover - telemetry must not break flow
        return None


def reset_drift_metric() -> None:
    """Drop the cached drift counter (test hygiene)."""
    global _drift_counter
    _drift_counter = None


def record_catalogue_drift_metric(drift: "CatalogueDrift") -> None:
    """Emit the catalogue-drift metric (best-effort, RouteIQ-9a42).

    Records the live drift count on the ``gateway.bedrock.catalogue_drift``
    counter, split by direction (``missing`` / ``extra``), so a CloudWatch alarm
    (or any Prometheus query) can fire on a non-zero deployed-vs-offered gap.
    Telemetry must never raise: a missing meter (OTel disabled) is a no-op.
    """
    try:
        counter = _get_drift_counter()
        if counter is None:
            return
        if drift.missing:
            counter.add(len(drift.missing), {"direction": "missing"})
        if drift.extra:
            counter.add(len(drift.extra), {"direction": "extra"})
    except Exception:  # pragma: no cover - telemetry must not break flow
        pass


# ============================================================================
# Settings glue
# ============================================================================


def _get_discovery_settings() -> "BedrockDiscoverySettings":
    """Lazily fetch the nested discovery settings from the gateway singleton.

    Imported lazily to avoid a settings import side effect at module import.
    """
    from litellm_llmrouter.settings import get_settings

    return get_settings().bedrock_discovery
