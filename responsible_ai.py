"""Responsible-AI helpers for V14 decision-support messaging and metrics."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def get_numeric_score(analysis: Dict[str, Any]) -> Optional[float]:
    candidates = [
        analysis.get("ats_score"),
        (analysis.get("ats") or {}).get("score") if isinstance(analysis.get("ats"), dict) else None,
        (analysis.get("explainability") or {}).get("ats_score") if isinstance(analysis.get("explainability"), dict) else None,
    ]
    for value in candidates:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def evaluate_threshold(
    rows: Iterable[Dict[str, Any]],
    labels: Dict[str, str],
    threshold: float,
) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    considered = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        label = labels.get(candidate_id)
        if label not in {"qualified", "not_qualified"}:
            continue
        score = get_numeric_score(row.get("analysis") or {})
        if score is None:
            continue
        predicted = score >= threshold
        actual = label == "qualified"
        considered += 1
        if predicted and actual:
            tp += 1
        elif not predicted and not actual:
            tn += 1
        elif predicted and not actual:
            fp += 1
        else:
            fn += 1

    accuracy = ((tp + tn) / considered * 100.0) if considered else None
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) else None
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    actual_qualified = tp + fn
    actual_not_qualified = tn + fp
    return {
        "sample_size": considered,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_qualified": tp + fp,
        "actual_qualified": actual_qualified,
        "actual_not_qualified": actual_not_qualified,
    }


RESPONSIBLE_AI_NOTICE = (
    "Decision-support tooling only: scores and detected requirements are signals for recruiter review. "
    "A missing signal does not prove a candidate lacks a skill. Keep human review and job-relevant evidence in the loop."
)
