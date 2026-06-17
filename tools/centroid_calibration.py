"""Semantic-intent centroid calibration helper (RouteIQ-44a1).

A STANDALONE operator utility — it does NOT import or modify the routing
strategy class (``centroid_routing.CentroidClassifier``). It answers one
question an operator needs before trusting centroid-based semantic-intent
routing:

    "Are my intent centroids actually SEPARATED, or do they overlap so much
     that the classifier is effectively guessing?"

Centroid routing classifies a prompt by cosine similarity to a set of
pre-computed, unit-normalized centroid vectors (e.g. ``simple`` vs ``complex``,
or per-intent centroids). If two centroids point in nearly the same direction
(cosine similarity near 1.0) the decision boundary between them is razor-thin
and noise dominates — the router cannot reliably tell the buckets apart. This
tool reports the INTER-CENTROID COSINE SEPARATION so an operator can catch that
at config-load time instead of in production.

Two input modes (mutually exclusive):

  1. ``--centroid-dir DIR``  — load the shipped ``*_centroid.npy`` vectors from a
     directory (the same files ``centroid_routing.CentroidClassifier`` loads) and
     report pairwise separation. No model, no network.

  2. ``--exemplars FILE``    — an operator supplies their OWN candidate intent
     buckets as exemplar phrases (a JSON ``{bucket: [phrase, ...]}`` map). The
     tool embeds them (via the optional ``sentence-transformers`` embedder),
     builds one mean centroid per bucket, and reports the separation — letting an
     operator A/B a proposed intent taxonomy BEFORE generating shipped centroids.

The CORE math (``cosine_separation`` / ``calibrate``) is pure-numpy and fully
unit-testable WITHOUT any embedder, npy file, or AWS credential — tests pass
vectors directly. The embedder is imported lazily only on the ``--exemplars``
path so the import and the ``--centroid-dir`` path stay dependency-light.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

# A "weak" separation floor below which two centroids are flagged as too close
# to discriminate reliably. Cosine separation = 1 - cosine_similarity, so a
# separation of 0.0 means identical direction (un-discriminable) and ~2.0 means
# opposite. 0.10 is a conservative, documented heuristic (overridable via
# ``--min-separation``), NOT a hard correctness oracle.
DEFAULT_MIN_SEPARATION = 0.10


@dataclass
class CentroidPair:
    """The cosine relationship between two named centroids."""

    a: str
    b: str
    cosine_similarity: float
    cosine_separation: float  # 1 - cosine_similarity
    well_separated: bool


@dataclass
class CalibrationReport:
    """Inter-centroid separation report for a set of named centroids."""

    names: list[str]
    min_separation: float  # the threshold used to flag weak pairs
    pairs: list[CentroidPair] = field(default_factory=list)
    weak_pairs: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool | None:
        """True when every pair clears the separation floor, False when any pair
        is weak, None when there are fewer than two centroids to compare."""
        if len(self.names) < 2:
            return None
        return not self.weak_pairs

    @property
    def worst(self) -> CentroidPair | None:
        """The closest (least separated) pair — the calibration smoking gun."""
        if not self.pairs:
            return None
        return min(self.pairs, key=lambda p: p.cosine_separation)


def _require_numpy() -> Any:
    try:
        import numpy as np  # noqa: PLC0415 — lazy, optional dep
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "numpy is required for centroid calibration; `uv sync` installs it."
        ) from exc
    return np


def _normalize(vec: Any) -> Any:
    """Return a unit-normalized copy of ``vec`` (a zero vector passes through)."""
    np = _require_numpy()
    arr = np.asarray(vec, dtype="float64")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def cosine_separation(a: Any, b: Any) -> tuple[float, float]:
    """Return ``(cosine_similarity, cosine_separation)`` for two vectors.

    Both are normalized first, so the similarity is the dot product of the unit
    vectors. ``cosine_separation = 1 - cosine_similarity`` (0 == same direction,
    1 == orthogonal, 2 == opposite). Pure numpy — no embedder, no I/O.
    """
    np = _require_numpy()
    ua = _normalize(a)
    ub = _normalize(b)
    sim = float(np.dot(ua, ub))
    # Guard tiny float drift outside [-1, 1].
    sim = max(-1.0, min(1.0, sim))
    return sim, 1.0 - sim


def calibrate(
    centroids: dict[str, Any],
    *,
    min_separation: float = DEFAULT_MIN_SEPARATION,
) -> CalibrationReport:
    """Compute pairwise inter-centroid cosine separation (RouteIQ-44a1).

    ``centroids`` maps a bucket/intent name to its centroid vector (any array-
    like; normalized internally). Every unordered pair is scored; a pair whose
    separation falls below ``min_separation`` is flagged ``weak`` (too close to
    discriminate reliably). Deterministic pair order (sorted by name) so the
    report is stable. Never raises on a single centroid — the report just has no
    pairs and ``healthy=None``.
    """
    names = sorted(centroids)
    report = CalibrationReport(names=names, min_separation=min_separation)
    if len(names) < 2:
        report.notes.append(
            "Fewer than two centroids supplied; nothing to compare "
            "(separation is only meaningful between buckets)."
        )
        return report
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sim, sep = cosine_separation(centroids[a], centroids[b])
            well = sep >= min_separation
            report.pairs.append(
                CentroidPair(
                    a=a,
                    b=b,
                    cosine_similarity=round(sim, 6),
                    cosine_separation=round(sep, 6),
                    well_separated=well,
                )
            )
            if not well:
                report.weak_pairs.append((a, b))
    return report


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------


def load_centroid_dir(centroid_dir: str) -> dict[str, Any]:
    """Load every ``*_centroid.npy`` vector in ``centroid_dir`` into a name->vec
    map (the bucket name is the filename minus the ``_centroid.npy`` suffix).

    Mirrors the files ``centroid_routing.CentroidClassifier`` loads
    (``simple_centroid.npy`` / ``complex_centroid.npy``) without importing the
    strategy class. Raises ``FileNotFoundError`` when the directory holds no
    centroid files.
    """
    np = _require_numpy()
    if not os.path.isdir(centroid_dir):
        raise FileNotFoundError(f"centroid dir not found: {centroid_dir}")
    out: dict[str, Any] = {}
    for fname in sorted(os.listdir(centroid_dir)):
        if not fname.endswith("_centroid.npy"):
            continue
        name = fname[: -len("_centroid.npy")]
        out[name] = np.load(os.path.join(centroid_dir, fname))
    if not out:
        raise FileNotFoundError(
            f"no *_centroid.npy files in {centroid_dir} "
            "(expected e.g. simple_centroid.npy / complex_centroid.npy)"
        )
    return out


def centroids_from_exemplars(
    exemplars: dict[str, Sequence[str]],
    *,
    embed: Any = None,
) -> dict[str, Any]:
    """Build one mean centroid per bucket from operator exemplar phrases.

    ``exemplars`` maps a bucket name to a list of example phrases that should
    route to that bucket. ``embed`` is a ``list[str] -> ndarray`` callable
    (injectable for tests); when None it lazily constructs an
    ``all-MiniLM-L6-v2`` sentence-transformers embedder (the same family the
    shipped centroids use). Each bucket's centroid is the mean of its phrase
    embeddings, unit-normalized. Buckets with no phrases are skipped.
    """
    np = _require_numpy()
    if embed is None:
        embed = _default_embedder()
    out: dict[str, Any] = {}
    for name, phrases in exemplars.items():
        phrase_list = [p for p in phrases if isinstance(p, str) and p.strip()]
        if not phrase_list:
            continue
        vectors = np.asarray(embed(phrase_list), dtype="float64")
        mean = vectors.mean(axis=0)
        out[name] = _normalize(mean)
    return out


def _default_embedder() -> Any:  # pragma: no cover - needs the optional model
    """Lazily build an ``all-MiniLM-L6-v2`` embedder callable.

    Imported only on the ``--exemplars`` path so neither the import nor the
    ``--centroid-dir`` path depends on sentence-transformers / torch.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(phrases: Sequence[str]) -> Any:
        return model.encode(list(phrases), normalize_embeddings=False)

    return _embed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: CalibrationReport) -> str:
    """Render a human-readable calibration report (stdout-friendly)."""
    lines: list[str] = []
    lines.append("=== RouteIQ semantic-intent centroid calibration ===")
    lines.append(f"centroids        : {', '.join(report.names) or '<none>'}")
    lines.append(f"min separation   : {report.min_separation:.4f}")
    if report.healthy is None:
        lines.append("verdict          : NOT ASSESSABLE (need >= 2 centroids)")
    elif report.healthy:
        lines.append("verdict          : HEALTHY (all pairs well-separated)")
    else:
        weak = ", ".join(f"{a}~{b}" for a, b in report.weak_pairs)
        lines.append(f"verdict          : WEAK SEPARATION ({weak})")
    lines.append("")
    lines.append("pairwise cosine separation (1 - cos_sim; higher == better):")
    for p in sorted(report.pairs, key=lambda q: q.cosine_separation):
        flag = "" if p.well_separated else "  <-- TOO CLOSE"
        lines.append(
            f"  {p.a:<18} ~ {p.b:<18} "
            f"sep={p.cosine_separation:.4f}  (cos={p.cosine_similarity:+.4f}){flag}"
        )
    worst = report.worst
    if worst is not None:
        lines.append("")
        lines.append(
            f"closest pair     : {worst.a} ~ {worst.b} "
            f"(separation {worst.cosine_separation:.4f})"
        )
    for note in report.notes:
        lines.append(f"note             : {note}")
    return "\n".join(lines)


def report_to_dict(report: CalibrationReport) -> dict[str, Any]:
    """JSON-serializable view of the calibration report (for tooling)."""
    return {
        "centroids": report.names,
        "min_separation": report.min_separation,
        "healthy": report.healthy,
        "weak_pairs": [list(pair) for pair in report.weak_pairs],
        "pairs": [
            {
                "a": p.a,
                "b": p.b,
                "cosine_similarity": p.cosine_similarity,
                "cosine_separation": p.cosine_separation,
                "well_separated": p.well_separated,
            }
            for p in report.pairs
        ],
        "notes": report.notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="routeiq-centroid-calibration",
        description=(
            "Report inter-centroid cosine separation for RouteIQ semantic-intent "
            "centroid routing. Flags centroid pairs too close to discriminate "
            "reliably. Standalone: does not touch the routing strategy class."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--centroid-dir",
        default=None,
        help="Directory of shipped *_centroid.npy vectors (e.g. models/centroids).",
    )
    src.add_argument(
        "--exemplars",
        default=None,
        help="JSON file mapping bucket -> [exemplar phrase, ...] to embed and "
        "calibrate (needs sentence-transformers).",
    )
    p.add_argument(
        "--min-separation",
        type=float,
        default=DEFAULT_MIN_SEPARATION,
        help=f"Cosine-separation floor for a pair to be 'well separated' "
        f"(default {DEFAULT_MIN_SEPARATION}).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of the human-readable table.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.centroid_dir:
        centroids = load_centroid_dir(args.centroid_dir)
    else:
        with open(args.exemplars, encoding="utf-8") as fh:
            exemplars = json.load(fh)
        if not isinstance(exemplars, dict):
            print(
                "exemplars file must be a JSON object {bucket: [phrase]}",
                file=sys.stderr,
            )
            return 2
        centroids = centroids_from_exemplars(exemplars)
    report = calibrate(centroids, min_separation=args.min_separation)
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_report(report))
    # Non-zero exit when separation is weak so it can gate a config-load check.
    return 1 if report.healthy is False else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
