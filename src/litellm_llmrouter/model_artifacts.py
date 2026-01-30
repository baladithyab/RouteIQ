"""
Model Artifact Verification Module
==================================

This module provides cryptographic verification of model artifacts (especially pickle files)
to prevent arbitrary code execution from malicious model files.

Security Features:
- SHA256 hash verification against a signed manifest
- Optional Ed25519 signature verification of manifest
- Safe activation with rollback on verification failure

Usage:
    manifest_path = os.environ.get("LLMROUTER_MODEL_MANIFEST_PATH")
    verifier = ModelArtifactVerifier(manifest_path)

    # Before loading a pickle model:
    verifier.verify_artifact("models/knn_router.pkl")  # Raises ModelVerificationError on failure
"""

import base64
import hashlib
import hmac
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger

# Try to import cryptography for Ed25519 signature verification
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Try to import yaml for YAML manifest support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ModelVerificationError(Exception):
    """Raised when model artifact verification fails."""

    def __init__(
        self,
        artifact_path: str,
        reason: str,
        details: Optional[Dict] = None,
    ):
        self.artifact_path = artifact_path
        self.reason = reason
        self.details = details or {}
        super().__init__(f"Model verification failed for '{artifact_path}': {reason}")


class ManifestParseError(Exception):
    """Raised when the manifest file cannot be parsed."""

    def __init__(self, manifest_path: str, reason: str):
        self.manifest_path = manifest_path
        self.reason = reason
        super().__init__(f"Failed to parse manifest '{manifest_path}': {reason}")


class SignatureVerificationError(Exception):
    """Raised when manifest signature verification fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Manifest signature verification failed: {reason}")


class SignatureType(Enum):
    """Supported signature types for manifest verification."""

    ED25519 = "ed25519"
    HMAC_SHA256 = "hmac-sha256"
    NONE = "none"


@dataclass
class ArtifactEntry:
    """Represents a single artifact entry in the manifest."""

    path: str
    sha256: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactEntry":
        """Create an ArtifactEntry from a dictionary."""
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            description=data.get("description"),
            created_at=data.get("created_at"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ModelManifest:
    """
    Represents a model artifact manifest with hash and signature information.

    Example manifest (JSON):
    {
        "version": "1.0",
        "created_at": "2024-01-28T10:00:00Z",
        "signature_type": "ed25519",
        "signature": "base64-encoded-signature",
        "artifacts": [
            {
                "path": "models/knn_router.pkl",
                "sha256": "abc123...",
                "description": "KNN router model v1.2",
                "created_at": "2024-01-28T09:00:00Z",
                "tags": ["production", "v1.2"]
            }
        ]
    }
    """

    version: str
    artifacts: List[ArtifactEntry]
    created_at: Optional[str] = None
    signature_type: SignatureType = SignatureType.NONE
    signature: Optional[str] = None  # Base64-encoded signature
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, path: str) -> Optional[ArtifactEntry]:
        """Get an artifact entry by path (supports relative path matching)."""
        # Normalize path for comparison
        normalized = Path(path).as_posix()
        for artifact in self.artifacts:
            artifact_normalized = Path(artifact.path).as_posix()
            if artifact_normalized == normalized or artifact_normalized.endswith(
                normalized
            ):
                return artifact
            # Also try matching the basename
            if Path(artifact.path).name == Path(path).name:
                return artifact
        return None

    def get_signable_content(self) -> bytes:
        """Get the canonical content to sign/verify (excludes signature field)."""
        content = {
            "version": self.version,
            "artifacts": [
                {"path": a.path, "sha256": a.sha256}
                for a in sorted(self.artifacts, key=lambda x: x.path)
            ],
        }
        if self.created_at:
            content["created_at"] = self.created_at
        if self.description:
            content["description"] = self.description
        # Use sorted keys for deterministic output
        return json.dumps(content, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        """Create a ModelManifest from a dictionary."""
        sig_type_str = data.get("signature_type", "none").lower()
        try:
            sig_type = SignatureType(sig_type_str)
        except ValueError:
            sig_type = SignatureType.NONE

        artifacts = [ArtifactEntry.from_dict(a) for a in data.get("artifacts", [])]

        return cls(
            version=data.get("version", "1.0"),
            artifacts=artifacts,
            created_at=data.get("created_at"),
            signature_type=sig_type,
            signature=data.get("signature"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ActiveModelVersion:
    """Tracks the currently active model version for observability."""

    artifact_path: str
    sha256: str
    manifest_path: Optional[str]
    loaded_at: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact_path": self.artifact_path,
            "sha256": self.sha256,
            "manifest_path": self.manifest_path,
            "loaded_at": self.loaded_at,
            "tags": self.tags,
        }


class ModelArtifactVerifier:
    """
    Verifies model artifacts against a signed manifest before loading.

    This class provides:
    - SHA256 hash verification of artifact files
    - Optional Ed25519 or HMAC-SHA256 signature verification of the manifest
    - Thread-safe manifest caching and reloading

    Environment Variables:
        LLMROUTER_MODEL_MANIFEST_PATH: Path to the manifest file (JSON or YAML)
        LLMROUTER_ENFORCE_SIGNED_MODELS: If "true", require manifest for pickle models
        LLMROUTER_MODEL_PUBLIC_KEY_B64: Base64-encoded Ed25519 public key
        LLMROUTER_MODEL_PUBLIC_KEY_PATH: Path to Ed25519 public key file
        LLMROUTER_MODEL_HMAC_SECRET: Secret for HMAC-SHA256 verification
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        public_key_b64: Optional[str] = None,
        public_key_path: Optional[str] = None,
        hmac_secret: Optional[str] = None,
        enforce_signed: Optional[bool] = None,
    ):
        """
        Initialize the verifier.

        Args:
            manifest_path: Path to manifest file, or from LLMROUTER_MODEL_MANIFEST_PATH
            public_key_b64: Base64-encoded Ed25519 public key, or from LLMROUTER_MODEL_PUBLIC_KEY_B64
            hmac_secret: HMAC secret, or from LLMROUTER_MODEL_HMAC_SECRET
            enforce_signed: Whether to enforce manifest verification, or from LLMROUTER_ENFORCE_SIGNED_MODELS
        """
        self.manifest_path = manifest_path or os.environ.get(
            "LLMROUTER_MODEL_MANIFEST_PATH"
        )
        self.public_key_b64 = public_key_b64 or os.environ.get(
            "LLMROUTER_MODEL_PUBLIC_KEY_B64"
        )
        self.public_key_path = public_key_path or os.environ.get(
            "LLMROUTER_MODEL_PUBLIC_KEY_PATH"
        )
        self.hmac_secret = hmac_secret or os.environ.get("LLMROUTER_MODEL_HMAC_SECRET")

        # Determine enforcement mode
        if enforce_signed is not None:
            self.enforce_signed = enforce_signed
        else:
            env_enforce = os.environ.get(
                "LLMROUTER_ENFORCE_SIGNED_MODELS", "false"
            ).lower()
            self.enforce_signed = env_enforce == "true"

        self._manifest: Optional[ModelManifest] = None
        self._manifest_lock = threading.RLock()
        self._manifest_mtime: float = 0
        self._public_key: Optional[Any] = None  # Ed25519PublicKey

        # Track active model versions for observability
        self._active_versions: Dict[str, ActiveModelVersion] = {}
        self._versions_lock = threading.RLock()

    def _load_public_key(self) -> Optional[Any]:
        """Load the Ed25519 public key from config."""
        if not CRYPTOGRAPHY_AVAILABLE:
            verbose_proxy_logger.warning(
                "cryptography package not available; Ed25519 signature verification disabled"
            )
            return None

        if self._public_key is not None:
            return self._public_key

        key_data: Optional[bytes] = None

        # Try base64-encoded key first
        if self.public_key_b64:
            try:
                key_data = base64.b64decode(self.public_key_b64)
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Failed to decode public key from base64: {e}"
                )
                return None

        # Try file path
        elif self.public_key_path:
            try:
                with open(self.public_key_path, "rb") as f:
                    key_data = f.read()
            except Exception as e:
                verbose_proxy_logger.error(f"Failed to read public key file: {e}")
                return None

        if key_data is None:
            return None

        # Try to load as raw key (32 bytes) or PEM
        try:
            if len(key_data) == 32:
                # Raw 32-byte key
                self._public_key = ed25519.Ed25519PublicKey.from_public_bytes(key_data)
            else:
                # Try PEM format
                self._public_key = serialization.load_pem_public_key(key_data)
            return self._public_key
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load Ed25519 public key: {e}")
            return None

    def _load_manifest(self, force_reload: bool = False) -> Optional[ModelManifest]:
        """Load and cache the manifest file."""
        if not self.manifest_path:
            return None

        with self._manifest_lock:
            # Check if we need to reload
            if not force_reload and self._manifest is not None:
                try:
                    current_mtime = Path(self.manifest_path).stat().st_mtime
                    if current_mtime <= self._manifest_mtime:
                        return self._manifest
                except OSError:
                    pass

            # Load manifest
            path = Path(self.manifest_path)
            if not path.exists():
                verbose_proxy_logger.warning(
                    f"Manifest file not found: {self.manifest_path}"
                )
                return None

            try:
                with open(path, "r") as f:
                    content = f.read()

                # Parse based on file extension
                if path.suffix.lower() in (".yaml", ".yml"):
                    if not YAML_AVAILABLE:
                        raise ManifestParseError(
                            self.manifest_path, "YAML format requires pyyaml package"
                        )
                    data = yaml.safe_load(content)
                else:
                    data = json.loads(content)

                manifest = ModelManifest.from_dict(data)
                self._manifest = manifest
                self._manifest_mtime = path.stat().st_mtime

                verbose_proxy_logger.info(
                    f"Loaded model manifest: {self.manifest_path} "
                    f"(version={manifest.version}, artifacts={len(manifest.artifacts)})"
                )
                return manifest

            except (json.JSONDecodeError, yaml.YAMLError) as e:
                raise ManifestParseError(self.manifest_path, str(e))
            except Exception as e:
                verbose_proxy_logger.error(f"Failed to load manifest: {e}")
                return None

    def verify_manifest_signature(self, manifest: ModelManifest) -> bool:
        """
        Verify the signature on a manifest.

        Returns True if:
        - Signature type is 'none' (no signature required)
        - Signature is valid

        Raises SignatureVerificationError if signature is invalid.
        """
        if manifest.signature_type == SignatureType.NONE:
            return True

        if not manifest.signature:
            raise SignatureVerificationError(
                "Manifest has signature_type but no signature"
            )

        signable_content = manifest.get_signable_content()
        signature_bytes = base64.b64decode(manifest.signature)

        if manifest.signature_type == SignatureType.ED25519:
            if not CRYPTOGRAPHY_AVAILABLE:
                raise SignatureVerificationError(
                    "Ed25519 verification requires cryptography package"
                )

            public_key = self._load_public_key()
            if public_key is None:
                raise SignatureVerificationError(
                    "No public key configured for Ed25519 verification. "
                    "Set LLMROUTER_MODEL_PUBLIC_KEY_B64 or LLMROUTER_MODEL_PUBLIC_KEY_PATH"
                )

            try:
                public_key.verify(signature_bytes, signable_content)
                verbose_proxy_logger.debug(
                    "Manifest Ed25519 signature verified successfully"
                )
                return True
            except InvalidSignature:
                raise SignatureVerificationError(
                    "Ed25519 signature verification failed"
                )

        elif manifest.signature_type == SignatureType.HMAC_SHA256:
            if not self.hmac_secret:
                raise SignatureVerificationError(
                    "No HMAC secret configured. Set LLMROUTER_MODEL_HMAC_SECRET"
                )

            expected_sig = hmac.new(
                self.hmac_secret.encode("utf-8"), signable_content, hashlib.sha256
            ).digest()

            if not hmac.compare_digest(expected_sig, signature_bytes):
                raise SignatureVerificationError(
                    "HMAC-SHA256 signature verification failed"
                )

            verbose_proxy_logger.debug(
                "Manifest HMAC-SHA256 signature verified successfully"
            )
            return True

        else:
            raise SignatureVerificationError(
                f"Unknown signature type: {manifest.signature_type}"
            )

    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def verify_artifact(
        self,
        artifact_path: str,
        require_manifest: bool = False,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Verify an artifact against the manifest.

        Args:
            artifact_path: Path to the artifact file to verify
            require_manifest: If True, raise error if no manifest entry exists
            correlation_id: Optional request/correlation ID for logging

        Returns:
            True if verification passes

        Raises:
            ModelVerificationError: If verification fails
            FileNotFoundError: If artifact file doesn't exist
        """
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        # Load manifest
        manifest = self._load_manifest()

        # Check if manifest is required
        should_require = require_manifest or self.enforce_signed
        if manifest is None:
            if should_require:
                raise ModelVerificationError(
                    artifact_path,
                    "No manifest configured but verification is required",
                    {"correlation_id": correlation_id} if correlation_id else {},
                )
            verbose_proxy_logger.warning(
                f"{log_prefix}No manifest configured; skipping verification for {artifact_path}"
            )
            return True

        # Verify manifest signature if present
        try:
            self.verify_manifest_signature(manifest)
        except SignatureVerificationError as e:
            raise ModelVerificationError(
                artifact_path,
                f"Manifest signature verification failed: {e.reason}",
                {"correlation_id": correlation_id} if correlation_id else {},
            )

        # Find artifact in manifest
        entry = manifest.get_artifact(artifact_path)
        if entry is None:
            if should_require:
                raise ModelVerificationError(
                    artifact_path,
                    "Artifact not found in manifest",
                    (
                        {
                            "manifest_path": self.manifest_path,
                            "correlation_id": correlation_id,
                        }
                        if correlation_id
                        else {"manifest_path": self.manifest_path}
                    ),
                )
            verbose_proxy_logger.warning(
                f"{log_prefix}Artifact not in manifest; skipping verification for {artifact_path}"
            )
            return True

        # Compute and compare hash
        actual_hash = self.compute_sha256(artifact_path)
        if actual_hash != entry.sha256:
            raise ModelVerificationError(
                artifact_path,
                "SHA256 hash mismatch",
                {
                    "expected": entry.sha256,
                    "actual": actual_hash,
                    "manifest_path": self.manifest_path,
                    "correlation_id": correlation_id,
                },
            )

        verbose_proxy_logger.info(
            f"{log_prefix}Artifact verified: {artifact_path} (sha256={actual_hash[:16]}...)"
        )
        return True

    def record_active_version(
        self,
        artifact_path: str,
        sha256: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ActiveModelVersion:
        """
        Record an artifact as the active version for observability.

        Args:
            artifact_path: Path to the artifact
            sha256: Pre-computed hash (will be computed if not provided)
            tags: Optional tags for the version

        Returns:
            ActiveModelVersion instance
        """
        if sha256 is None:
            sha256 = self.compute_sha256(artifact_path)

        # Get tags from manifest if available
        manifest = self._load_manifest()
        manifest_tags = []
        if manifest:
            entry = manifest.get_artifact(artifact_path)
            if entry:
                manifest_tags = entry.tags

        version = ActiveModelVersion(
            artifact_path=artifact_path,
            sha256=sha256,
            manifest_path=self.manifest_path,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or manifest_tags,
        )

        with self._versions_lock:
            self._active_versions[artifact_path] = version

        verbose_proxy_logger.debug(
            f"Recorded active model version: {artifact_path} (sha256={sha256[:16]}...)"
        )
        return version

    def get_active_version(self, artifact_path: str) -> Optional[ActiveModelVersion]:
        """Get the active version info for an artifact."""
        with self._versions_lock:
            return self._active_versions.get(artifact_path)

    def get_all_active_versions(self) -> Dict[str, ActiveModelVersion]:
        """Get all active model versions."""
        with self._versions_lock:
            return dict(self._active_versions)


# Global verifier instance (lazy initialized)
_global_verifier: Optional[ModelArtifactVerifier] = None
_verifier_lock = threading.Lock()


def get_artifact_verifier() -> ModelArtifactVerifier:
    """Get or create the global artifact verifier instance."""
    global _global_verifier
    with _verifier_lock:
        if _global_verifier is None:
            _global_verifier = ModelArtifactVerifier()
        return _global_verifier


def verify_model_artifact(
    artifact_path: str,
    require_manifest: bool = False,
    correlation_id: Optional[str] = None,
) -> bool:
    """
    Convenience function to verify an artifact using the global verifier.

    Args:
        artifact_path: Path to the artifact file
        require_manifest: If True, fail if no manifest entry exists
        correlation_id: Optional correlation ID for logging

    Returns:
        True if verification passes

    Raises:
        ModelVerificationError: If verification fails
    """
    verifier = get_artifact_verifier()
    return verifier.verify_artifact(artifact_path, require_manifest, correlation_id)


def create_manifest_signer(
    private_key_b64: Optional[str] = None,
    private_key_path: Optional[str] = None,
    hmac_secret: Optional[str] = None,
) -> "ManifestSigner":
    """Create a manifest signer for generating signed manifests."""
    return ManifestSigner(private_key_b64, private_key_path, hmac_secret)


class ManifestSigner:
    """
    Utility class for creating signed manifests.

    This is typically used by the model training/deployment pipeline,
    not by the gateway itself.

    Example:
        signer = ManifestSigner(private_key_path="/path/to/key.pem")
        manifest = signer.create_manifest(
            artifacts=[
                {"path": "models/knn.pkl", "sha256": "abc123..."}
            ],
            signature_type=SignatureType.ED25519
        )
        signer.save_manifest(manifest, "models/manifest.json")
    """

    def __init__(
        self,
        private_key_b64: Optional[str] = None,
        private_key_path: Optional[str] = None,
        hmac_secret: Optional[str] = None,
    ):
        self.private_key_b64 = private_key_b64
        self.private_key_path = private_key_path
        self.hmac_secret = hmac_secret
        self._private_key: Optional[Any] = None

    def _load_private_key(self) -> Optional[Any]:
        """Load the Ed25519 private key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("cryptography package required for Ed25519 signing")

        if self._private_key is not None:
            return self._private_key

        key_data: Optional[bytes] = None

        if self.private_key_b64:
            key_data = base64.b64decode(self.private_key_b64)
        elif self.private_key_path:
            with open(self.private_key_path, "rb") as f:
                key_data = f.read()

        if key_data is None:
            raise ValueError("No private key configured")

        # Try to load as raw key (32 bytes) or PEM
        if len(key_data) == 32:
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_data)
        else:
            self._private_key = serialization.load_pem_private_key(
                key_data, password=None
            )

        return self._private_key

    def _sign_content(
        self, signable_content: bytes, signature_type: SignatureType
    ) -> str:
        pass

    def sign_manifest(
        self,
        manifest: ModelManifest,
        signature_type: SignatureType = SignatureType.ED25519,
    ) -> ModelManifest:
        """Sign a manifest and return a new manifest with the signature."""
        manifest.signature_type = signature_type
        signable_content = manifest.get_signable_content()

        if signature_type == SignatureType.ED25519:
            private_key = self._load_private_key()
            signature = private_key.sign(signable_content)
        elif signature_type == SignatureType.HMAC_SHA256:
            if not self.hmac_secret:
                raise ValueError("HMAC secret required for HMAC-SHA256 signing")
            signature = hmac.new(
                self.hmac_secret.encode("utf-8"), signable_content, hashlib.sha256
            ).digest()
        else:
            raise ValueError(f"Unsupported signature type: {signature_type}")

        manifest.signature = base64.b64encode(signature).decode("ascii")
        return manifest

    def create_manifest(
        self,
        artifacts: List[Union[Dict[str, Any], ArtifactEntry]],
        signature_type: SignatureType = SignatureType.NONE,
        version: str = "1.0",
        description: Optional[str] = None,
    ) -> ModelManifest:
        """
        Create a new manifest with optional signing.

        Args:
            artifacts: List of artifact entries (dicts or ArtifactEntry objects)
            signature_type: Type of signature to apply
            version: Manifest version string
            description: Optional description

        Returns:
            ModelManifest instance (signed if signature_type is not NONE)
        """
        entries = []
        for a in artifacts:
            if isinstance(a, ArtifactEntry):
                entries.append(a)
            else:
                entries.append(ArtifactEntry.from_dict(a))

        manifest = ModelManifest(
            version=version,
            artifacts=entries,
            created_at=datetime.now(timezone.utc).isoformat(),
            signature_type=signature_type,
            description=description,
        )

        if signature_type != SignatureType.NONE:
            manifest = self.sign_manifest(manifest, signature_type)

        return manifest

    def create_artifact_entry(
        self,
        file_path: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ArtifactEntry:
        """Create an artifact entry by computing the hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return ArtifactEntry(
            path=file_path,
            sha256=sha256_hash.hexdigest(),
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or [],
        )

    def save_manifest(
        self,
        manifest: ModelManifest,
        output_path: str,
        format: str = "json",
    ) -> None:
        """Save a manifest to a file."""
        data = {
            "version": manifest.version,
            "created_at": manifest.created_at,
            "signature_type": manifest.signature_type.value,
            "artifacts": [
                {
                    "path": a.path,
                    "sha256": a.sha256,
                    "description": a.description,
                    "created_at": a.created_at,
                    "tags": a.tags,
                    "metadata": a.metadata,
                }
                for a in manifest.artifacts
            ],
        }
        if manifest.signature:
            data["signature"] = manifest.signature
        if manifest.description:
            data["description"] = manifest.description
        if manifest.metadata:
            data["metadata"] = manifest.metadata

        with open(output_path, "w") as f:
            if format == "yaml" and YAML_AVAILABLE:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2, sort_keys=False)


def generate_ed25519_keypair() -> tuple[str, str]:
    """
    Generate a new Ed25519 keypair for manifest signing.

    Returns:
        Tuple of (private_key_b64, public_key_b64)
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError("cryptography package required for key generation")

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes_raw()
    public_bytes = public_key.public_bytes_raw()

    return (
        base64.b64encode(private_bytes).decode("ascii"),
        base64.b64encode(public_bytes).decode("ascii"),
    )
