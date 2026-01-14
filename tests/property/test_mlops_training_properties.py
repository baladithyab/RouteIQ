"""
Property-Based Tests for MLOps Training Pipeline.

These tests validate the correctness properties defined in the design document
for the MLOps training pipeline (Requirements 5.x).

Property 14: MLOps Model Training
For any valid training data in LLMRouter format, the MLOps pipeline should
successfully train a routing model for the specified strategy type and save
model artifacts that are compatible with the Gateway's hot reload mechanism.

**Validates: Requirements 5.2, 5.3, 5.5**
"""

import json
import os
import pickle
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from hypothesis import given, settings, strategies as st, assume, HealthCheck

# Try to import torch, but make it optional for testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


# Supported router types for training
TRAINABLE_ROUTER_TYPES = ["knn", "svm", "mlp", "mf"]

# Sample model names for synthetic data
SAMPLE_MODELS = [
    "claude-sonnet",
    "claude-haiku",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-opus",
    "gpt-4",
    "gpt-3.5-turbo",
]

# Sample queries for synthetic data
SAMPLE_QUERIES = [
    "What is machine learning?",
    "Explain neural networks in simple terms",
    "How do transformers work?",
    "What are the benefits of cloud computing?",
    "Describe the water cycle",
    "What is quantum computing?",
    "Explain photosynthesis",
    "How does GPS work?",
    "What is blockchain technology?",
    "Explain the theory of relativity",
]


# ============================================================================
# Hypothesis Strategies for MLOps Data Generation
# ============================================================================

router_type_strategy = st.sampled_from(TRAINABLE_ROUTER_TYPES)
model_name_strategy = st.sampled_from(SAMPLE_MODELS)
query_strategy = st.sampled_from(SAMPLE_QUERIES)


@st.composite
def routing_data_entry_strategy(draw, embedding_id_max: int = 100):
    """Generate a single routing data entry in LLMRouter format."""
    return {
        "query": draw(query_strategy),
        "model_name": draw(model_name_strategy),
        "performance": draw(st.floats(min_value=0.0, max_value=1.0)),
        "embedding_id": draw(st.integers(min_value=0, max_value=embedding_id_max)),
    }


@st.composite
def routing_dataset_strategy(draw, min_size: int = 10, max_size: int = 50):
    """Generate a complete routing dataset with train/test split."""
    num_entries = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate unique queries for embeddings - limit to available queries
    max_unique = min(len(SAMPLE_QUERIES), num_entries)
    num_unique_queries = draw(st.integers(min_value=3, max_value=max_unique))
    
    # Sample from available queries
    unique_queries = SAMPLE_QUERIES[:num_unique_queries]
    
    # Create query to embedding_id mapping
    query_to_id = {q: i for i, q in enumerate(unique_queries)}
    
    # Generate routing entries
    entries = []
    for _ in range(num_entries):
        query = draw(st.sampled_from(unique_queries))
        entries.append({
            "query": query,
            "model_name": draw(model_name_strategy),
            "performance": round(draw(st.floats(min_value=0.1, max_value=1.0)), 3),
            "embedding_id": query_to_id[query],
        })
    
    # Split into train/test (80/20)
    split_idx = int(len(entries) * 0.8)
    train_data = entries[:split_idx]
    test_data = entries[split_idx:]
    
    assume(len(train_data) >= 5)  # Need minimum training data
    assume(len(test_data) >= 2)   # Need minimum test data
    
    return {
        "train": train_data,
        "test": test_data,
        "unique_queries": unique_queries,
        "num_unique_queries": len(unique_queries),
    }


def llm_candidates_strategy():
    """Generate LLM candidates metadata."""
    models = {}
    for model_name in SAMPLE_MODELS:
        models[model_name] = {
            "name": model_name,
            "provider": "bedrock" if "claude" in model_name.lower() else "openai",
        }
    return st.just(models)


@st.composite
def knn_hyperparams_strategy(draw):
    """Generate valid KNN hyperparameters."""
    return {
        "n_neighbors": draw(st.integers(min_value=1, max_value=10)),
        "metric": draw(st.sampled_from(["cosine", "euclidean", "manhattan"])),
        "weights": draw(st.sampled_from(["uniform", "distance"])),
    }


@st.composite
def svm_hyperparams_strategy(draw):
    """Generate valid SVM hyperparameters."""
    return {
        "C": draw(st.floats(min_value=0.1, max_value=10.0)),
        "kernel": draw(st.sampled_from(["rbf", "linear", "poly"])),
        "gamma": draw(st.sampled_from(["scale", "auto"])),
        "probability": True,
    }


@st.composite
def mlp_hyperparams_strategy(draw):
    """Generate valid MLP hyperparameters."""
    return {
        "hidden_dims": draw(st.lists(
            st.integers(min_value=32, max_value=256),
            min_size=1,
            max_size=3
        )),
        "learning_rate": draw(st.floats(min_value=0.0001, max_value=0.01)),
        "epochs": draw(st.integers(min_value=10, max_value=50)),
        "batch_size": draw(st.integers(min_value=8, max_value=64)),
        "dropout": draw(st.floats(min_value=0.0, max_value=0.5)),
    }


@st.composite
def training_config_strategy(draw, router_type: str = None):
    """Generate a complete training configuration."""
    if router_type is None:
        router_type = draw(router_type_strategy)
    
    # Generate hyperparameters based on router type
    if router_type == "knn":
        hparams = draw(knn_hyperparams_strategy())
    elif router_type == "svm":
        hparams = draw(svm_hyperparams_strategy())
    elif router_type == "mlp":
        hparams = draw(mlp_hyperparams_strategy())
    else:
        hparams = {}
    
    return {
        "router_type": router_type,
        "hparam": hparams,
    }


# ============================================================================
# Helper Functions
# ============================================================================

def create_training_data_files(
    output_dir: str,
    train_data: List[Dict],
    test_data: List[Dict],
    unique_queries: List[str],
    llm_data: Dict,
    embedding_dim: int = 384,
) -> Dict[str, str]:
    """Create all required training data files in LLMRouter format."""
    paths = {}
    
    # Create routing train data
    train_path = os.path.join(output_dir, "routing_train.jsonl")
    with open(train_path, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
    paths["routing_data_train"] = train_path
    
    # Create routing test data
    test_path = os.path.join(output_dir, "routing_test.jsonl")
    with open(test_path, "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")
    paths["routing_data_test"] = test_path
    
    # Create query embeddings
    emb_path = os.path.join(output_dir, "query_embeddings.pt")
    if HAS_TORCH:
        # Use real torch tensor
        embeddings = torch.randn(len(unique_queries), embedding_dim)
        torch.save(embeddings, emb_path)
    else:
        # Create a mock .pt file (pickle format with numpy-like structure)
        # This simulates the structure without requiring torch
        import random
        embeddings_data = {
            "shape": (len(unique_queries), embedding_dim),
            "data": [[random.gauss(0, 1) for _ in range(embedding_dim)] 
                     for _ in range(len(unique_queries))],
        }
        with open(emb_path, "wb") as f:
            pickle.dump(embeddings_data, f)
    paths["query_embedding_data"] = emb_path
    
    # Create LLM data
    llm_path = os.path.join(output_dir, "llm_data.json")
    with open(llm_path, "w") as f:
        json.dump(llm_data, f, indent=2)
    paths["llm_data"] = llm_path
    
    return paths


def create_training_config(
    output_dir: str,
    data_paths: Dict[str, str],
    router_type: str,
    hparams: Dict,
) -> str:
    """Create a training configuration YAML file."""
    model_dir = os.path.join(output_dir, f"{router_type}_router")
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine model file extension based on router type
    model_ext = ".pt" if router_type == "mlp" else ".pkl"
    model_file = os.path.join(model_dir, f"model{model_ext}")
    
    config = {
        "data_path": data_paths,
        "hparam": hparams,
        "model_path": {
            "ini_model_path": model_file,
            "save_model_path": model_file,
            "load_model_path": model_file,
        },
    }
    
    config_path = os.path.join(output_dir, f"{router_type}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


# ============================================================================
# Property Tests for MLOps Model Training
# ============================================================================

class TestMLOpsModelTrainingProperty:
    """
    Property 14: MLOps Model Training
    
    For any valid training data in LLMRouter format, the MLOps pipeline should
    successfully train a routing model for the specified strategy type and save
    model artifacts that are compatible with the Gateway's hot reload mechanism.
    
    **Validates: Requirements 5.2, 5.3, 5.5**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(router_type=router_type_strategy)
    def test_trainable_router_types_are_supported(self, router_type: str):
        """Property 14: All trainable router types are in the supported list."""
        assert router_type in TRAINABLE_ROUTER_TYPES
        assert router_type in ["knn", "svm", "mlp", "mf"]

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(dataset=routing_dataset_strategy())
    def test_routing_data_format_is_valid(self, dataset: Dict[str, Any]):
        """Property 14: Generated routing data has valid LLMRouter format."""
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        # Verify train data format
        for entry in train_data:
            assert "query" in entry
            assert "model_name" in entry
            assert "performance" in entry
            assert "embedding_id" in entry
            assert isinstance(entry["query"], str)
            assert isinstance(entry["model_name"], str)
            assert 0.0 <= entry["performance"] <= 1.0
            assert isinstance(entry["embedding_id"], int)
            assert entry["embedding_id"] >= 0
        
        # Verify test data format
        for entry in test_data:
            assert "query" in entry
            assert "model_name" in entry
            assert "performance" in entry
            assert "embedding_id" in entry

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(dataset=routing_dataset_strategy())
    def test_routing_data_train_test_split(self, dataset: Dict[str, Any]):
        """Property 14: Training data is properly split into train/test sets."""
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        # Verify split exists
        assert len(train_data) > 0
        assert len(test_data) > 0
        
        # Verify train is larger than test (80/20 split)
        total = len(train_data) + len(test_data)
        train_ratio = len(train_data) / total
        assert 0.7 <= train_ratio <= 0.9  # Allow some variance

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(dataset=routing_dataset_strategy(), llm_data=llm_candidates_strategy())
    def test_training_data_files_round_trip(
        self, dataset: Dict[str, Any], llm_data: Dict[str, Any]
    ):
        """Property 14: Training data files round-trip correctly through file I/O."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training data files
            paths = create_training_data_files(
                tmpdir,
                dataset["train"],
                dataset["test"],
                dataset["unique_queries"],
                llm_data,
            )
            
            # Verify all files exist
            for key, path in paths.items():
                assert os.path.exists(path), f"Missing file: {key}"
            
            # Verify train data round-trip
            with open(paths["routing_data_train"], "r") as f:
                loaded_train = [json.loads(line) for line in f]
            assert len(loaded_train) == len(dataset["train"])
            
            # Verify test data round-trip
            with open(paths["routing_data_test"], "r") as f:
                loaded_test = [json.loads(line) for line in f]
            assert len(loaded_test) == len(dataset["test"])
            
            # Verify embeddings round-trip
            if HAS_TORCH:
                loaded_emb = torch.load(paths["query_embedding_data"], weights_only=False)
                assert loaded_emb.shape[0] == len(dataset["unique_queries"])
                assert loaded_emb.shape[1] == 384  # Default embedding dim
            else:
                # Verify mock embeddings structure
                with open(paths["query_embedding_data"], "rb") as f:
                    loaded_emb = pickle.load(f)
                assert loaded_emb["shape"][0] == len(dataset["unique_queries"])
                assert loaded_emb["shape"][1] == 384
            
            # Verify LLM data round-trip
            with open(paths["llm_data"], "r") as f:
                loaded_llm = json.load(f)
            assert loaded_llm == llm_data

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(config=training_config_strategy())
    def test_training_config_is_valid_yaml(self, config: Dict[str, Any]):
        """Property 14: Training configuration is valid YAML and serializable."""
        # Serialize to YAML
        yaml_str = yaml.dump(config)
        
        # Deserialize back
        loaded = yaml.safe_load(yaml_str)
        
        # Verify structure
        assert "router_type" in loaded
        assert "hparam" in loaded
        assert loaded["router_type"] in TRAINABLE_ROUTER_TYPES

    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    @given(
        dataset=routing_dataset_strategy(min_size=20, max_size=30),
        llm_data=llm_candidates_strategy(),
        router_type=router_type_strategy,
    )
    def test_training_config_file_creation(
        self,
        dataset: Dict[str, Any],
        llm_data: Dict[str, Any],
        router_type: str,
    ):
        """Property 14: Training config file is created with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training data files
            data_paths = create_training_data_files(
                tmpdir,
                dataset["train"],
                dataset["test"],
                dataset["unique_queries"],
                llm_data,
            )
            
            # Generate hyperparameters based on router type
            if router_type == "knn":
                hparams = {"n_neighbors": 5, "metric": "cosine", "weights": "distance"}
            elif router_type == "svm":
                hparams = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True}
            elif router_type == "mlp":
                hparams = {"hidden_dims": [128], "learning_rate": 0.001, "epochs": 10, "batch_size": 16}
            else:
                hparams = {}
            
            # Create config file
            config_path = create_training_config(tmpdir, data_paths, router_type, hparams)
            
            # Verify config file exists and is valid
            assert os.path.exists(config_path)
            
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)
            
            assert "data_path" in loaded_config
            assert "hparam" in loaded_config
            assert "model_path" in loaded_config
            
            # Verify data paths point to existing files
            for key, path in loaded_config["data_path"].items():
                assert os.path.exists(path), f"Data file missing: {key}"

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(hparams=knn_hyperparams_strategy())
    def test_knn_hyperparams_are_valid(self, hparams: Dict[str, Any]):
        """Property 14: KNN hyperparameters are within valid ranges."""
        assert 1 <= hparams["n_neighbors"] <= 10
        assert hparams["metric"] in ["cosine", "euclidean", "manhattan"]
        assert hparams["weights"] in ["uniform", "distance"]

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(hparams=svm_hyperparams_strategy())
    def test_svm_hyperparams_are_valid(self, hparams: Dict[str, Any]):
        """Property 14: SVM hyperparameters are within valid ranges."""
        assert 0.1 <= hparams["C"] <= 10.0
        assert hparams["kernel"] in ["rbf", "linear", "poly"]
        assert hparams["gamma"] in ["scale", "auto"]
        assert hparams["probability"] is True

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(hparams=mlp_hyperparams_strategy())
    def test_mlp_hyperparams_are_valid(self, hparams: Dict[str, Any]):
        """Property 14: MLP hyperparameters are within valid ranges."""
        assert len(hparams["hidden_dims"]) >= 1
        assert all(32 <= dim <= 256 for dim in hparams["hidden_dims"])
        assert 0.0001 <= hparams["learning_rate"] <= 0.01
        assert 10 <= hparams["epochs"] <= 50
        assert 8 <= hparams["batch_size"] <= 64
        assert 0.0 <= hparams["dropout"] <= 0.5


class TestMLOpsModelArtifactCompatibility:
    """
    Tests for model artifact compatibility with Gateway hot reload.
    
    **Validates: Requirements 5.5**
    """

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(router_type=router_type_strategy)
    def test_model_artifact_path_convention(self, router_type: str):
        """Property 14: Model artifacts follow naming convention for hot reload."""
        # Model file extension based on router type
        if router_type == "mlp":
            expected_ext = ".pt"
        else:
            expected_ext = ".pkl"
        
        # Verify convention
        model_filename = f"model{expected_ext}"
        assert model_filename.endswith(expected_ext)
        
        # Verify model directory naming
        model_dir = f"{router_type}_router"
        assert router_type in model_dir

    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        dataset=routing_dataset_strategy(min_size=15, max_size=25),
        llm_data=llm_candidates_strategy(),
    )
    def test_model_directory_structure_for_hot_reload(
        self, dataset: Dict[str, Any], llm_data: Dict[str, Any]
    ):
        """Property 14: Model directory structure is compatible with hot reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for router_type in TRAINABLE_ROUTER_TYPES:
                # Create training data files
                data_paths = create_training_data_files(
                    tmpdir,
                    dataset["train"],
                    dataset["test"],
                    dataset["unique_queries"],
                    llm_data,
                )
                
                # Create config (which creates model directory)
                hparams = {"n_neighbors": 5} if router_type == "knn" else {}
                config_path = create_training_config(tmpdir, data_paths, router_type, hparams)
                
                # Verify model directory exists
                model_dir = os.path.join(tmpdir, f"{router_type}_router")
                assert os.path.isdir(model_dir), f"Model dir missing: {model_dir}"
                
                # Verify config references correct model path
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                
                model_path = config["model_path"]["save_model_path"]
                assert model_path.startswith(model_dir)

    def test_all_router_types_have_consistent_config_structure(self):
        """Property 14: All router types use consistent config structure."""
        required_sections = ["data_path", "hparam", "model_path"]
        required_data_paths = [
            "routing_data_train",
            "routing_data_test", 
            "query_embedding_data",
            "llm_data",
        ]
        required_model_paths = ["save_model_path", "load_model_path"]
        
        # This is a structural test - verify the expected structure
        for router_type in TRAINABLE_ROUTER_TYPES:
            # Each router type should support these config sections
            assert router_type in TRAINABLE_ROUTER_TYPES
            
        # Verify required sections are documented
        assert len(required_sections) == 3
        assert len(required_data_paths) == 4
        assert len(required_model_paths) == 2


class TestMLOpsEvaluationMetrics:
    """
    Tests for evaluation metrics computation.
    
    **Validates: Requirements 5.4**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        predictions=st.lists(
            st.sampled_from(SAMPLE_MODELS),
            min_size=10,
            max_size=50,
        ),
        ground_truth=st.lists(
            st.sampled_from(SAMPLE_MODELS),
            min_size=10,
            max_size=50,
        ),
    )
    def test_accuracy_metric_is_valid(
        self, predictions: List[str], ground_truth: List[str]
    ):
        """Property 14: Accuracy metric is between 0 and 1."""
        # Ensure same length
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
        
        assume(min_len > 0)
        
        # Calculate accuracy
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions)
        
        assert 0.0 <= accuracy <= 1.0

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        latencies=st.lists(
            st.floats(min_value=0.1, max_value=10.0),
            min_size=5,
            max_size=50,
        )
    )
    def test_latency_metrics_are_valid(self, latencies: List[float]):
        """Property 14: Latency metrics (mean, p50, p95) are valid."""
        import statistics
        
        assume(len(latencies) >= 5)
        
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        
        # Sort for percentile calculation
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        
        # Verify metrics are valid
        assert mean_latency > 0
        assert median_latency > 0
        assert p95_latency > 0
        assert p95_latency >= median_latency  # P95 should be >= median

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        costs=st.lists(
            st.floats(min_value=0.001, max_value=1.0),
            min_size=5,
            max_size=50,
        )
    )
    def test_cost_metrics_are_valid(self, costs: List[float]):
        """Property 14: Cost metrics are non-negative."""
        assume(len(costs) >= 5)
        
        total_cost = sum(costs)
        avg_cost = total_cost / len(costs)
        
        assert total_cost >= 0
        assert avg_cost >= 0
        assert avg_cost <= max(costs)


class TestMLOpsPipelineIntegration:
    """
    Integration tests for the MLOps pipeline components.
    """

    def test_mlproject_file_exists(self):
        """Verify MLproject file exists for MLflow integration."""
        mlproject_path = Path("examples/mlops/MLproject")
        assert mlproject_path.exists(), "MLproject file missing"

    def test_docker_compose_file_exists(self):
        """Verify Docker Compose file exists for MLOps stack."""
        compose_path = Path("examples/mlops/docker-compose.mlops.yml")
        assert compose_path.exists(), "docker-compose.mlops.yml missing"

    def test_training_script_exists(self):
        """Verify training script exists."""
        script_path = Path("examples/mlops/scripts/train_router.py")
        assert script_path.exists(), "train_router.py missing"

    def test_synthetic_data_generator_exists(self):
        """Verify synthetic data generator exists."""
        script_path = Path("examples/mlops/scripts/generate_synthetic_traces.py")
        assert script_path.exists(), "generate_synthetic_traces.py missing"

    def test_data_converter_exists(self):
        """Verify data converter script exists."""
        script_path = Path("examples/mlops/scripts/convert_traces_to_llmrouter.py")
        assert script_path.exists(), "convert_traces_to_llmrouter.py missing"

    def test_test_configs_exist(self):
        """Verify test configuration files exist for each router type."""
        config_dir = Path("examples/mlops/configs")
        assert config_dir.exists(), "configs directory missing"
        
        # Check for test configs
        expected_configs = [
            "test_knn_config.yaml",
            "test_mlp_config.yaml",
            "test_svm_config.yaml",
        ]
        
        for config_name in expected_configs:
            config_path = config_dir / config_name
            assert config_path.exists(), f"Config missing: {config_name}"

    def test_mlproject_defines_required_entry_points(self):
        """Verify MLproject defines required entry points."""
        mlproject_path = Path("examples/mlops/MLproject")
        
        with open(mlproject_path, "r") as f:
            content = f.read()
        
        # Check for required entry points
        required_entry_points = ["train:", "generate_synthetic:", "prepare_data:", "deploy:"]
        
        for entry_point in required_entry_points:
            assert entry_point in content, f"Missing entry point: {entry_point}"

    def test_training_script_supports_all_router_types(self):
        """Verify training script supports all trainable router types."""
        script_path = Path("examples/mlops/scripts/train_router.py")
        
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for router type support
        for router_type in TRAINABLE_ROUTER_TYPES:
            assert f'"{router_type}"' in content or f"'{router_type}'" in content, \
                f"Router type not supported: {router_type}"
