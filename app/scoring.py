"""Deterministic goal score in [0, 1]. The LLM never guesses this number.

score = w_r * retrieval_confidence + w_p * provenance_coverage + w_f * fix_quality
where fix_quality = 1 / (1 + validator_fixes), so fewer repairs means a higher score.
Weights are normalised, so any positive weights keep the result in [0, 1].
"""
from __future__ import annotations

from typing import Optional

from app.config import settings


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_score(
    retrieval_confidence: float,
    provenance_coverage: float,
    validator_fixes: int,
    weights: Optional[tuple[float, float, float]] = None,
) -> float:
    w_r, w_p, w_f = weights or (
        settings.score_w_retrieval,
        settings.score_w_provenance,
        settings.score_w_fixes,
    )
    total = w_r + w_p + w_f
    if total <= 0:
        raise ValueError("score weights must sum to a positive number")
    fix_quality = 1.0 / (1.0 + max(0, int(validator_fixes)))
    raw = (
        w_r * _clamp(retrieval_confidence)
        + w_p * _clamp(provenance_coverage)
        + w_f * fix_quality
    ) / total
    return round(_clamp(raw), 4)
