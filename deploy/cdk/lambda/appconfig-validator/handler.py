"""AppConfig validator Lambda for the RouteIQ gateway config (config.yaml).

This is the REAL validator asset, bundled into the function zip by
``ConfigStateConstruct._build_validator_lambda`` via ``Code.from_asset`` when
Docker is available on the synth host. When Docker is absent (CI, isolated unit
tests, the cred-free synth gate) the construct falls back to an inline accept-all
placeholder instead (``index.lambda_handler``); this file's entry point is
``handler.lambda_handler`` on the bundled path.

Contract (identical to the vllm-sr-on-aws validator + AppConfig's own contract):
  * AppConfig invokes this function with the candidate configuration in the
    event payload and signals SUCCESS when the function RETURNS NORMALLY and
    FAILURE when it RAISES. So a valid config => ``return None``; an invalid
    config => raise (the raised message surfaces in the AppConfig deployment
    error and blocks the rollout).

Packaging note (cold-read / self-contained):
  This handler does NOT import ``litellm_llmrouter`` at module top. That package
  is NOT bundled into the validator zip (``requirements.txt`` pins PyYAML only).
  Instead it re-implements the *rules* of RouteIQ's config validation inline:
    (a) the candidate must parse as YAML and be a mapping (the same structural
        gate RouteIQ's loader applies);
    (b) RouteIQ-shape check: a ``model_list`` (or ``general:``/``router:``
        block) must be present and well-formed, mirroring the keys the gateway
        expects (``model_list`` + ``litellm_settings`` + a ``general:`` block);
    (c) inline-secret deny: raw provider keys / AWS keys / bearer tokens MUST
        NOT appear literally in the config (RouteIQ analogue of vllm-sr's F-29).
        RouteIQ configs reference secrets indirectly via ``os.environ/<VAR>``,
        so a literal ``sk-...`` / ``AKIA...`` value is a deploy-time rejection.
  This mirrors the way the vllm-sr handler.py is standalone (PyYAML + schema +
  inline-secret rules), so the asset stays self-contained and the rules track
  RouteIQ's loader/``GatewaySettings`` shape without importing the gateway.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any

import yaml

_LOG = logging.getLogger()
_LOG.setLevel(logging.INFO)


# Inline-secret deny patterns (RouteIQ analogue of vllm-sr F-29). RouteIQ
# configs reference secrets indirectly (``api_key: os.environ/ANTHROPIC_API_KEY``
# or ``aws_region_name: os.environ/AWS_REGION``), so a LITERAL provider/cloud
# credential in the config body is always a rejection. Patterns are deliberately
# conservative (anchored on the well-known credential prefixes) to avoid false
# positives on legitimate ``os.environ/...`` references.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # OpenAI / OpenAI-compatible / Anthropic-style secret keys: ``sk-...`` /
    # ``sk-ant-...`` with enough entropy to be a real key (>= 20 trailing chars).
    ("openai_or_anthropic_secret_key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}")),
    # AWS long-term access key id.
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # AWS ASIA (temporary) access key id.
    ("aws_temp_access_key_id", re.compile(r"\bASIA[0-9A-Z]{16}\b")),
    # An explicit aws_secret_access_key assignment carrying a literal value
    # (not an ``os.environ/`` reference).
    (
        "aws_secret_access_key_literal",
        re.compile(r"aws_secret_access_key\s*[:=]\s*(?!os\.environ/)['\"]?[A-Za-z0-9/+]{20,}"),
    ),
    # A literal bearer token in a header / authorization value.
    ("bearer_token", re.compile(r"[Bb]earer\s+[A-Za-z0-9._-]{20,}")),
)


def _decode_candidate(event: dict[str, Any]) -> str:
    """Return the candidate config as text from the AppConfig invocation event.

    AppConfig passes the candidate configuration to a Lambda validator in the
    event ``content`` field, base64-encoded. We tolerate both the encoded form
    and a plain-text fallback so the handler is robust to local test harnesses
    that pass the body directly.
    """
    content = event.get("content")
    if content is None:
        raise ValueError("AppConfig validator event missing 'content' field.")
    if isinstance(content, bytes):
        raw = content
    elif isinstance(content, str):
        try:
            raw = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError):
            # Not base64 (e.g. a local test passing raw YAML) - use as-is.
            return content
    else:
        raise ValueError(f"Unexpected 'content' type: {type(content).__name__}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Candidate config is not valid UTF-8.") from exc


def _reject_inline_secrets(text: str) -> None:
    """Raise if a literal credential is embedded in the config body (F-29)."""
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            # Do NOT echo the matched value - that would log the secret. Only
            # the rule label is surfaced (PII / secret-scrubbing posture).
            raise ValueError(
                f"Inline secret rejected: candidate config matched the "
                f"'{label}' deny rule. RouteIQ configs must reference secrets "
                f"indirectly via os.environ/<VAR>, never inline."
            )


def _validate_routeiq_shape(parsed: Any) -> None:
    """Structural RouteIQ-config shape check (loader / GatewaySettings analogue).

    Mirrors the keys the gateway's loader expects without importing it:
      * the document must be a mapping;
      * a ``model_list`` (list of mapping entries each with a ``model_name``)
        must be present, OR a top-level ``general:``/``router:`` block must be
        present (a config that only steers routing/governance with no model
        list is still well-formed for the gateway);
      * ``litellm_settings`` / ``general_settings`` / ``general`` / ``router``
        blocks, when present, must be mappings.
    """
    if not isinstance(parsed, dict):
        raise ValueError(
            "Candidate config must be a YAML mapping at the top level, "
            f"got {type(parsed).__name__}."
        )

    model_list = parsed.get("model_list")
    has_general = any(k in parsed for k in ("general", "general_settings", "router"))

    if model_list is None and not has_general:
        raise ValueError(
            "Candidate config has neither a 'model_list' nor a "
            "'general'/'general_settings'/'router' block; nothing for the "
            "RouteIQ gateway to load."
        )

    if model_list is not None:
        if not isinstance(model_list, list):
            raise ValueError("'model_list' must be a list.")
        for idx, entry in enumerate(model_list):
            if not isinstance(entry, dict):
                raise ValueError(f"model_list[{idx}] must be a mapping.")
            if not entry.get("model_name"):
                raise ValueError(f"model_list[{idx}] is missing 'model_name'.")
            params = entry.get("litellm_params")
            if params is not None and not isinstance(params, dict):
                raise ValueError(f"model_list[{idx}].litellm_params must be a mapping.")

    for block in ("litellm_settings", "general_settings", "general", "router"):
        value = parsed.get(block)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"'{block}' must be a mapping when present.")


def lambda_handler(event: dict[str, Any], context: Any) -> None:
    """AppConfig Lambda validator entry point.

    Returns ``None`` on success; raises on any validation failure (AppConfig's
    accept-by-return / reject-by-raise contract).
    """
    text = _decode_candidate(event)

    # (a) structural YAML parse.
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Candidate config is not valid YAML: {exc}") from exc

    # (b) inline-secret deny (run on the raw text before shape checks so a
    #     secret embedded anywhere is caught even if the shape is otherwise OK).
    _reject_inline_secrets(text)

    # (c) RouteIQ-shape validation.
    _validate_routeiq_shape(parsed)

    _LOG.info("RouteIQ AppConfig validator: candidate config accepted.")
    return None
