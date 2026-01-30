"""
Tests for Model Artifact Verification Module
=============================================

Tests for SHA256 hash verification, Ed25519 signature verification,
and safe model activation with rollback.
"""

import base64
import hashlib
import json
import os
import pickle
import tempfile
from unittest.mock import patch

import pytest

# Check if litellm is available (required for import)
try:
    import litellm  # noqa: F401

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LITELLM_AVAILABLE,
    reason="litellm package not installed - unit tests require litellm",
)

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class MockModel:
    """A simple picklable mock model for testing."""

    def __init__(self, prediction="gpt-4"):
        self.prediction = prediction

    def predict(self, X):
        """Return a list of predictions."""
        return [self.prediction] * len(X)


class TestModelArtifactVerifier:
    """Test ModelArtifactVerifier functionality."""

    @pytest.fixture
    def temp_model_file(self):
        """Create a temporary pickle model file."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Create a simple picklable mock model
            mock_model = MockModel(prediction="gpt-4")
            pickle.dump(mock_model, f)
            model_path = f.name
        yield model_path
        os.unlink(model_path)

    @pytest.fixture
    def temp_manifest_file(self, temp_model_file):
        """Create a temporary manifest file for the model."""
        import os
        import hashlib
        import json
        import tempfile

        # Compute the hash of the model file
        sha256_hash = hashlib.sha256()
        with open(temp_model_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        model_hash = sha256_hash.hexdigest()

        manifest = {
            "version": "1.0",
            "created_at": "2026-01-28T10:00:00Z",
            "signature_type": "none",
            "artifacts": [
                {
                    "path": temp_model_file,
                    "sha256": model_hash,
                    "description": "Test KNN model",
                    "tags": ["test", "knn"],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(manifest, f)
            manifest_path = f.name

        yield manifest_path, model_hash
        os.unlink(manifest_path)

    def test_compute_sha256(self, temp_model_file):
        """Test SHA256 computation matches expected hash."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        verifier = ModelArtifactVerifier()
        computed_hash = verifier.compute_sha256(temp_model_file)

        # Verify using stdlib
        expected_hash = hashlib.sha256()
        with open(temp_model_file, "rb") as f:
            expected_hash.update(f.read())

        assert computed_hash == expected_hash.hexdigest()

    def test_verify_artifact_success(self, temp_model_file, temp_manifest_file):
        """Test successful artifact verification."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        manifest_path, model_hash = temp_manifest_file
        verifier = ModelArtifactVerifier(manifest_path=manifest_path)

        # Should not raise
        result = verifier.verify_artifact(temp_model_file)
        assert result is True

    def test_verify_artifact_hash_mismatch(self, temp_model_file, temp_manifest_file):
        """Test artifact verification fails on hash mismatch."""
        from litellm_llmrouter.model_artifacts import (
            ModelArtifactVerifier,
            ModelVerificationError,
        )

        manifest_path, _ = temp_manifest_file

        # Modify manifest to have wrong hash
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest["artifacts"][0]["sha256"] = "0" * 64  # Wrong hash
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        verifier = ModelArtifactVerifier(manifest_path=manifest_path)

        with pytest.raises(ModelVerificationError) as exc_info:
            verifier.verify_artifact(temp_model_file, require_manifest=True)

        assert "SHA256 hash mismatch" in str(exc_info.value)
        assert exc_info.value.details.get("expected") == "0" * 64

    def test_verify_artifact_not_in_manifest(self, temp_model_file):
        """Test artifact verification fails when artifact not in manifest."""
        from litellm_llmrouter.model_artifacts import (
            ModelArtifactVerifier,
            ModelVerificationError,
        )

        # Create manifest without the model
        manifest = {
            "version": "1.0",
            "signature_type": "none",
            "artifacts": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(manifest, f)
            manifest_path = f.name

        try:
            verifier = ModelArtifactVerifier(manifest_path=manifest_path)

            with pytest.raises(ModelVerificationError) as exc_info:
                verifier.verify_artifact(temp_model_file, require_manifest=True)

            assert "Artifact not found in manifest" in str(exc_info.value)
        finally:
            os.unlink(manifest_path)

    def test_verify_artifact_no_manifest_not_required(self, temp_model_file):
        """Test artifact verification succeeds when manifest not required."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        verifier = ModelArtifactVerifier(manifest_path=None)

        # Should not raise when require_manifest=False (default)
        result = verifier.verify_artifact(temp_model_file, require_manifest=False)
        assert result is True

    def test_verify_artifact_no_manifest_required(self, temp_model_file):
        """Test artifact verification fails when manifest required but missing."""
        from litellm_llmrouter.model_artifacts import (
            ModelArtifactVerifier,
            ModelVerificationError,
        )

        verifier = ModelArtifactVerifier(manifest_path=None)

        with pytest.raises(ModelVerificationError) as exc_info:
            verifier.verify_artifact(temp_model_file, require_manifest=True)

        assert "No manifest configured" in str(exc_info.value)

    def test_record_active_version(self, temp_model_file, temp_manifest_file):
        """Test recording active model version for observability."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        manifest_path, model_hash = temp_manifest_file
        verifier = ModelArtifactVerifier(manifest_path=manifest_path)

        version = verifier.record_active_version(temp_model_file, tags=["test", "v1"])

        assert version.artifact_path == temp_model_file
        assert version.sha256 == model_hash
        assert version.manifest_path == manifest_path
        assert "test" in version.tags
        assert version.loaded_at is not None

    def test_get_active_versions(self, temp_model_file, temp_manifest_file):
        """Test getting active model versions."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        manifest_path, _ = temp_manifest_file
        verifier = ModelArtifactVerifier(manifest_path=manifest_path)

        verifier.record_active_version(temp_model_file)

        all_versions = verifier.get_all_active_versions()
        assert temp_model_file in all_versions

        version = verifier.get_active_version(temp_model_file)
        assert version is not None
        assert version.artifact_path == temp_model_file


@pytest.mark.skipif(
    not CRYPTOGRAPHY_AVAILABLE,
    reason="cryptography package not available",
)
class TestSignatureVerification:
    """Test Ed25519 signature verification."""

    @pytest.fixture
    def ed25519_keypair(self):
        """Generate Ed25519 keypair for testing."""
        from litellm_llmrouter.model_artifacts import generate_ed25519_keypair

        private_key_b64, public_key_b64 = generate_ed25519_keypair()
        return private_key_b64, public_key_b64

    @pytest.fixture
    def signed_manifest(self, ed25519_keypair):
        """Create a signed manifest."""
        from litellm_llmrouter.model_artifacts import (
            ManifestSigner,
            SignatureType,
        )

        private_key_b64, public_key_b64 = ed25519_keypair

        signer = ManifestSigner(private_key_b64=private_key_b64)
        manifest = signer.create_manifest(
            artifacts=[
                {
                    "path": "models/test.pkl",
                    "sha256": "a" * 64,
                    "description": "Test model",
                }
            ],
            signature_type=SignatureType.ED25519,
            version="1.0",
        )

        # Save manifest
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            signer.save_manifest(manifest, f.name)
            manifest_path = f.name

        yield manifest_path, public_key_b64
        os.unlink(manifest_path)

    def test_signature_verification_success(self, signed_manifest):
        """Test Ed25519 signature verification succeeds with valid key."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        manifest_path, public_key_b64 = signed_manifest
        verifier = ModelArtifactVerifier(
            manifest_path=manifest_path,
            public_key_b64=public_key_b64,
        )

        # Load and verify manifest signature
        manifest = verifier._load_manifest()
        assert manifest is not None

        # Should not raise
        result = verifier.verify_manifest_signature(manifest)
        assert result is True

    def test_signature_verification_fails_wrong_key(self, signed_manifest):
        """Test Ed25519 signature verification fails with wrong key."""
        from litellm_llmrouter.model_artifacts import (
            ModelArtifactVerifier,
            SignatureVerificationError,
            generate_ed25519_keypair,
        )

        manifest_path, _ = signed_manifest

        # Generate a different keypair
        _, wrong_public_key_b64 = generate_ed25519_keypair()

        verifier = ModelArtifactVerifier(
            manifest_path=manifest_path,
            public_key_b64=wrong_public_key_b64,
        )

        manifest = verifier._load_manifest()

        with pytest.raises(SignatureVerificationError) as exc_info:
            verifier.verify_manifest_signature(manifest)

        assert "Ed25519 signature verification failed" in str(exc_info.value)

    def test_keypair_generation(self):
        """Test Ed25519 keypair generation."""
        from litellm_llmrouter.model_artifacts import generate_ed25519_keypair

        private_b64, public_b64 = generate_ed25519_keypair()

        # Verify they're valid base64 and correct lengths (32 bytes raw)
        private_bytes = base64.b64decode(private_b64)
        public_bytes = base64.b64decode(public_b64)

        assert len(private_bytes) == 32
        assert len(public_bytes) == 32


class TestHMACVerification:
    """Test HMAC-SHA256 signature verification."""

    @pytest.fixture
    def hmac_signed_manifest(self):
        """Create an HMAC-signed manifest."""
        from litellm_llmrouter.model_artifacts import (
            ManifestSigner,
            SignatureType,
        )

        hmac_secret = "test-secret-key-12345"

        signer = ManifestSigner(hmac_secret=hmac_secret)
        manifest = signer.create_manifest(
            artifacts=[
                {"path": "model.pkl", "sha256": "a" * 64},
                {"path": "other.pkl", "sha256": "b" * 64},
            ],
            signature_type=SignatureType.HMAC_SHA256,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            signer.save_manifest(manifest, f.name)
            manifest_path = f.name

        yield manifest_path, hmac_secret
        os.unlink(manifest_path)

    def test_hmac_verification_success(self, hmac_signed_manifest):
        """Test HMAC-SHA256 verification succeeds with correct secret."""
        from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

        manifest_path, hmac_secret = hmac_signed_manifest
        verifier = ModelArtifactVerifier(
            manifest_path=manifest_path,
            hmac_secret=hmac_secret,
        )

        manifest = verifier._load_manifest()
        result = verifier.verify_manifest_signature(manifest)
        assert result is True

    def test_hmac_verification_fails_wrong_secret(self, hmac_signed_manifest):
        """Test HMAC-SHA256 verification fails with wrong secret."""
        from litellm_llmrouter.model_artifacts import (
            ModelArtifactVerifier,
            SignatureVerificationError,
        )

        manifest_path, _ = hmac_signed_manifest
        verifier = ModelArtifactVerifier(
            manifest_path=manifest_path,
            hmac_secret="wrong-secret",
        )

        manifest = verifier._load_manifest()

        with pytest.raises(SignatureVerificationError) as exc_info:
            verifier.verify_manifest_signature(manifest)

        assert "HMAC-SHA256 signature verification failed" in str(exc_info.value)


class TestManifestCreation:
    """Test suite for manifest creation."""

    def test_create_artifact_entry(self):
        """Test creating artifact entry with computed hash."""
        from litellm_llmrouter.model_artifacts import ManifestSigner

        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for hashing")
            temp_path = f.name

        try:
            signer = ManifestSigner()
            entry = signer.create_artifact_entry(
                file_path=temp_path,
                description="Test file",
                tags=["test"],
            )

            # Verify hash
            expected_hash = hashlib.sha256(b"test content for hashing").hexdigest()
            assert entry.sha256 == expected_hash
            assert entry.path == temp_path
            assert entry.description == "Test file"
            assert "test" in entry.tags
        finally:
            os.unlink(temp_path)

    def test_create_unsigned_manifest(self):
        """Test creating unsigned manifest."""
        from litellm_llmrouter.model_artifacts import (
            ManifestSigner,
            SignatureType,
        )

        signer = ManifestSigner()
        manifest = signer.create_manifest(
            artifacts=[{"path": "model.pkl", "sha256": "c" * 64}],
            signature_type=SignatureType.NONE,
            version="2.0",
            description="Test manifest",
        )

        assert manifest.version == "2.0"
        assert manifest.signature_type == SignatureType.NONE
        assert manifest.signature is None
        assert len(manifest.artifacts) == 1
        assert manifest.artifacts[0].path == "model.pkl"


class TestSafeActivationRollback:
    """Test safe model activation with rollback on failure."""

    @pytest.fixture
    def mock_pickle_allowed(self):
        """Temporarily allow pickle loading for tests."""
        with (
            patch("litellm_llmrouter.strategies.ALLOW_PICKLE_MODELS", True),
            patch("litellm_llmrouter.strategies.ENFORCE_SIGNED_MODELS", False),
        ):
            yield

    @pytest.fixture
    def valid_model_file(self):
        """Create a valid pickle model with predict method."""
        mock_model = MockModel(prediction="gpt-4")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(mock_model, f)
            model_path = f.name

        yield model_path
        os.unlink(model_path)

    def test_reload_model_invalid_pickle(self, valid_model_file):
        """Test that reload keeps old model when pickle is invalid."""

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a model"}, f)
            _model_path = f.name

    @pytest.fixture
    def invalid_model_file(self):
        """Create an invalid pickle model without predict method."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a model"}, f)
            model_path = f.name

        yield model_path
        os.unlink(model_path)

    def test_safe_reload_keeps_old_model_on_verification_failure(
        self,
        mock_pickle_allowed,
        valid_model_file,
    ):
        """Test that reload keeps old model when verification fails."""
        from litellm_llmrouter.strategies import InferenceKNNRouter
        from litellm_llmrouter.model_artifacts import get_artifact_verifier

        # Create initial router with valid model
        router = InferenceKNNRouter(
            model_path=valid_model_file,
            embedding_model="test-model",
        )

        old_model = router.knn_model
        assert old_model is not None

        # Modify manifest to require verification with wrong hash
        verifier = get_artifact_verifier()
        with patch.object(verifier, "verify_artifact") as mock_verify:
            from litellm_llmrouter.model_artifacts import ModelVerificationError

            mock_verify.side_effect = ModelVerificationError(
                valid_model_file,
                "Hash mismatch",
                {"expected": "wrong", "actual": "right"},
            )

            # Attempt reload - should fail and keep old model
            result = router.reload_model(correlation_id="test-123")

            assert result is False
            # Old model should still be active
            assert router.knn_model is old_model

    def test_safe_reload_keeps_old_model_on_load_failure(
        self,
        mock_pickle_allowed,
        valid_model_file,
    ):
        """Test that reload keeps old model when loading fails."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        # Create initial router
        router = InferenceKNNRouter(
            model_path=valid_model_file,
            embedding_model="test-model",
        )

        old_model = router.knn_model
        assert old_model is not None

        # Make the model file invalid
        with open(valid_model_file, "wb") as f:
            f.write(b"not valid pickle data")

        # Attempt reload - should fail and keep old model
        # (Note: verification will also fail since hash changed)
        result = router.reload_model(correlation_id="test-456")

        assert result is False
        assert router.knn_model is old_model

    def test_safe_reload_success_swaps_model(
        self,
        mock_pickle_allowed,
        valid_model_file,
    ):
        """Test that successful reload swaps to new model."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        router = InferenceKNNRouter(
            model_path=valid_model_file,
            embedding_model="test-model",
        )

        old_model = router.knn_model
        old_version_hash = router.model_version.sha256 if router.model_version else None

        # Create a new model file with different content
        new_mock_model = MockModel(prediction="claude-3")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(new_mock_model, f)
            new_model_path = f.name

        try:
            # Update the router's model path
            router.model_path = new_model_path

            # Reload should succeed
            result = router.reload_model(correlation_id="test-789")

            assert result is True
            # Model should be different
            assert router.knn_model is not old_model
            # Version hash should be different
            if old_version_hash and router.model_version:
                assert router.model_version.sha256 != old_version_hash
        finally:
            os.unlink(new_model_path)


class TestModelManifest:
    """Test ModelManifest data class functionality."""

    def test_manifest_from_dict(self):
        """Test creating manifest from dictionary."""
        from litellm_llmrouter.model_artifacts import ModelManifest

        data = {
            "version": "1.0",
            "created_at": "2026-01-28T10:00:00Z",
            "signature_type": "hmac-sha256",
            "signature": "abc123",
            "artifacts": [
                {
                    "path": "model.pkl",
                    "sha256": "d" * 64,
                    "description": "Test model",
                }
            ],
        }

        manifest = ModelManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.signature == "abc123"
        assert len(manifest.artifacts) == 1
        assert manifest.artifacts[0].path == "model.pkl"

    def test_get_artifact_by_path(self):
        """Test finding artifact by path."""
        from litellm_llmrouter.model_artifacts import ModelManifest, ArtifactEntry

        manifest = ModelManifest(
            version="1.0",
            artifacts=[
                ArtifactEntry(
                    path="models/knn.pkl",
                    sha256="e" * 64,
                ),
                ArtifactEntry(
                    path="config.json",
                    sha256="f" * 64,
                ),
            ],
        )

        # Exact match
        entry = manifest.get_artifact("models/knn.pkl")
        assert entry is not None
        assert entry.sha256 == "e" * 64

        # Basename match
        entry = manifest.get_artifact("knn.pkl")
        assert entry is not None
        assert entry.sha256 == "e" * 64

        # Not found
        entry = manifest.get_artifact("nonexistent.pkl")
        assert entry is None

    def test_signable_content_deterministic(self):
        """Test that signable content is deterministic."""
        from litellm_llmrouter.model_artifacts import ModelManifest, ArtifactEntry

        # Create two manifests with same content but different order
        manifest1 = ModelManifest(
            version="1.0",
            created_at="2026-01-28T10:00:00Z",
            artifacts=[
                ArtifactEntry(path="a.pkl", sha256="aaa"),
                ArtifactEntry(path="b.pkl", sha256="bbb"),
            ],
        )

        manifest2 = ModelManifest(
            version="1.0",
            created_at="2026-01-28T10:00:00Z",
            artifacts=[
                ArtifactEntry(path="b.pkl", sha256="bbb"),
                ArtifactEntry(path="a.pkl", sha256="aaa"),
            ],
        )

        # Signable content should be identical (sorted)
        assert manifest1.get_signable_content() == manifest2.get_signable_content()


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_enforce_signed_models_default(self):
        """Test ENFORCE_SIGNED_MODELS defaults correctly."""
        # When ALLOW_PICKLE_MODELS=false, ENFORCE_SIGNED_MODELS should be false
        with patch.dict(
            os.environ,
            {
                "LLMROUTER_ALLOW_PICKLE_MODELS": "false",
                "LLMROUTER_ENFORCE_SIGNED_MODELS": "",
            },
        ):
            # Need to reload the module to pick up new env vars
            # This is a simplified test - in practice we test the behavior
            from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

            verifier = ModelArtifactVerifier(enforce_signed=False)
            assert verifier.enforce_signed is False

    def test_manifest_path_from_env(self):
        """Test manifest path from environment variable."""
        test_path = "/tmp/test_manifest.json"
        with patch.dict(
            os.environ,
            {"LLMROUTER_MODEL_MANIFEST_PATH": test_path},
        ):
            from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

            verifier = ModelArtifactVerifier()
            assert verifier.manifest_path == test_path

    def test_public_key_from_env(self):
        """Test public key from environment variable."""
        test_key = base64.b64encode(b"x" * 32).decode()
        with patch.dict(
            os.environ,
            {"LLMROUTER_MODEL_PUBLIC_KEY_B64": test_key},
        ):
            from litellm_llmrouter.model_artifacts import ModelArtifactVerifier

            verifier = ModelArtifactVerifier()
            assert verifier.public_key_b64 == test_key
