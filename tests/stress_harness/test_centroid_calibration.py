"""Tests for the semantic-intent centroid calibration helper (RouteIQ-44a1).

The calibration tool is a STANDALONE operator utility that reports inter-centroid
cosine separation; it does NOT import or modify the routing strategy class. The
core math is pure-numpy and is exercised here by passing vectors directly — NO
embedder, NO npy file, NO AWS credential. The two input loaders
(``--centroid-dir`` / ``--exemplars``) are exercised against tmp files and an
injected fake embedder.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import centroid_calibration as cc


# --- core math: cosine separation -----------------------------------------


def test_cosine_separation_orthogonal_is_one():
    sim, sep = cc.cosine_separation(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert sim == pytest.approx(0.0, abs=1e-9)
    assert sep == pytest.approx(1.0, abs=1e-9)


def test_cosine_separation_identical_is_zero():
    sim, sep = cc.cosine_separation(np.array([2.0, 0.0]), np.array([5.0, 0.0]))
    assert sim == pytest.approx(1.0, abs=1e-9)
    assert sep == pytest.approx(0.0, abs=1e-9)


def test_cosine_separation_opposite_is_two():
    sim, sep = cc.cosine_separation(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert sim == pytest.approx(-1.0, abs=1e-9)
    assert sep == pytest.approx(2.0, abs=1e-9)


def test_cosine_separation_normalizes_inputs():
    # un-normalized vectors must give the same answer as normalized ones.
    sim_a, _ = cc.cosine_separation(np.array([3.0, 4.0]), np.array([4.0, 3.0]))
    sim_b, _ = cc.cosine_separation(np.array([0.6, 0.8]), np.array([0.8, 0.6]))
    assert sim_a == pytest.approx(sim_b, abs=1e-9)


# --- calibrate over a centroid set ----------------------------------------


def test_calibrate_well_separated_set_healthy():
    centroids = {
        "math": np.array([1.0, 0.0, 0.0]),
        "code": np.array([0.0, 1.0, 0.0]),
        "creative": np.array([0.0, 0.0, 1.0]),
    }
    report = cc.calibrate(centroids)
    assert report.healthy is True
    assert report.weak_pairs == []
    assert len(report.pairs) == 3  # C(3,2)


def test_calibrate_flags_weak_pair():
    centroids = {
        "math": np.array([1.0, 0.0]),
        # nearly identical to math -> tiny separation, flagged weak.
        "arithmetic": np.array([1.0, 0.01]),
        "creative": np.array([0.0, 1.0]),
    }
    report = cc.calibrate(centroids, min_separation=0.10)
    assert report.healthy is False
    assert ("arithmetic", "math") in report.weak_pairs
    worst = report.worst
    assert worst is not None
    assert {worst.a, worst.b} == {"math", "arithmetic"}


def test_calibrate_single_centroid_not_assessable():
    report = cc.calibrate({"only": np.array([1.0, 0.0])})
    assert report.healthy is None
    assert report.pairs == []
    assert report.notes  # explains why


def test_calibrate_pair_order_is_deterministic():
    centroids = {
        "zeta": np.array([1.0, 0.0]),
        "alpha": np.array([0.0, 1.0]),
    }
    report = cc.calibrate(centroids)
    assert report.names == ["alpha", "zeta"]
    assert report.pairs[0].a == "alpha"
    assert report.pairs[0].b == "zeta"


# --- exemplar-phrase input (operator A/B of a proposed taxonomy) ----------


def test_centroids_from_exemplars_with_injected_embedder():
    """Operator exemplar phrases -> one mean centroid per bucket, via an injected
    embedder (no sentence-transformers dependency in the test)."""

    def fake_embed(phrases):
        # deterministic toy embedding: map known phrases to fixed directions.
        table = {
            "what is 2+2": [1.0, 0.0],
            "solve for x": [1.0, 0.0],
            "write a poem": [0.0, 1.0],
            "tell me a story": [0.0, 1.0],
        }
        return np.array([table[p] for p in phrases], dtype="float64")

    exemplars = {
        "math": ["what is 2+2", "solve for x"],
        "creative": ["write a poem", "tell me a story"],
    }
    centroids = cc.centroids_from_exemplars(exemplars, embed=fake_embed)
    report = cc.calibrate(centroids)
    # math centroid ~ [1,0], creative ~ [0,1] -> orthogonal, well separated.
    assert report.healthy is True
    assert set(centroids) == {"math", "creative"}


def test_centroids_from_exemplars_skips_empty_bucket():
    def fake_embed(phrases):
        return np.array([[1.0, 0.0] for _ in phrases], dtype="float64")

    centroids = cc.centroids_from_exemplars(
        {"a": ["x"], "empty": [], "blank": ["   "]}, embed=fake_embed
    )
    assert set(centroids) == {"a"}


# --- the npy-dir loader ----------------------------------------------------


def test_load_centroid_dir_reads_npy_vectors(tmp_path):
    np.save(tmp_path / "simple_centroid.npy", np.array([1.0, 0.0, 0.0]))
    np.save(tmp_path / "complex_centroid.npy", np.array([0.0, 1.0, 0.0]))
    centroids = cc.load_centroid_dir(str(tmp_path))
    assert set(centroids) == {"simple", "complex"}
    report = cc.calibrate(centroids)
    assert report.healthy is True


def test_load_centroid_dir_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cc.load_centroid_dir(str(tmp_path))


# --- CLI + reporting -------------------------------------------------------


def test_cli_centroid_dir_human_report(tmp_path, capsys):
    np.save(tmp_path / "math_centroid.npy", np.array([1.0, 0.0]))
    np.save(tmp_path / "creative_centroid.npy", np.array([0.0, 1.0]))
    rc = cc.main(["--centroid-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "centroid calibration" in out
    assert "HEALTHY" in out
    assert "math" in out and "creative" in out


def test_cli_weak_separation_exits_nonzero(tmp_path, capsys):
    np.save(tmp_path / "a_centroid.npy", np.array([1.0, 0.0]))
    np.save(tmp_path / "b_centroid.npy", np.array([1.0, 0.001]))
    rc = cc.main(["--centroid-dir", str(tmp_path), "--min-separation", "0.10"])
    # weak separation -> non-zero exit so it can gate a config-load check.
    assert rc == 1
    assert "WEAK SEPARATION" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys):
    np.save(tmp_path / "x_centroid.npy", np.array([1.0, 0.0]))
    np.save(tmp_path / "y_centroid.npy", np.array([0.0, 1.0]))
    rc = cc.main(["--centroid-dir", str(tmp_path), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["healthy"] is True
    assert data["centroids"] == ["x", "y"]
    assert len(data["pairs"]) == 1


def test_cli_exemplars_file(tmp_path, monkeypatch, capsys):
    """--exemplars path: a JSON taxonomy file is embedded + calibrated. We inject
    a fake embedder so no sentence-transformers / torch is needed."""
    exemplars_file = tmp_path / "intents.json"
    exemplars_file.write_text(
        json.dumps(
            {
                "billing": ["how much do I owe", "update my card"],
                "support": ["my app crashed", "reset my password"],
            }
        )
    )

    def fake_embed(phrases):
        table = {
            "how much do I owe": [1.0, 0.0],
            "update my card": [1.0, 0.0],
            "my app crashed": [0.0, 1.0],
            "reset my password": [0.0, 1.0],
        }
        return np.array([table[p] for p in phrases], dtype="float64")

    monkeypatch.setattr(cc, "_default_embedder", lambda: fake_embed)
    rc = cc.main(["--exemplars", str(exemplars_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "billing" in out and "support" in out
    assert "HEALTHY" in out
