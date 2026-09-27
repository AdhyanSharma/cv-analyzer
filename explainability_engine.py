"""
Explainable Matching & Evidence Engine V9

Deterministic, evidence-first explanations for the AI Resume Screening Platform.

The engine does not create a new hiring score. It explains the existing
project-defined ATS components and connects them to requirement coverage and
verifiable resume text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ATS_WEIGHTS = {
    "Skill Match": 35.0,
    "Semantic Match": 30.0,
    "Lexical Match": 20.0,
    "Keyword Match": 15.0,
}

COMPONENT_TO_KEY = {
    "Skill Match": "skill_score",
    "Semantic Match": "semantic_score",
    "Lexical Match": "lexical_score",
    "Keyword Match": "keyword_score",
}


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "using", "with", "you", "your",
    "will", "we", "our", "they", "them", "but", "can", "should", "must",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values or []:
        item = _clean(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _flatten_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        items: List[str] = []
        for nested in value.values():
            items.extend(_flatten_values(nested))
        return items
    if isinstance(value, (list, tuple, set)):
        items = []
        for nested in value:
            items.extend(_flatten_values(nested))
        return items
    return [_clean(value)] if _clean(value) else []


def score_breakdown(
    ats_result: Dict[str, Any] | None,
    analysis_result: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return the existing ATS components with project-defined weights."""
    ats_result = ats_result or {}
    analysis_result = analysis_result or {}

    rows: List[Dict[str, Any]] = []
    for label, weight in ATS_WEIGHTS.items():
        key = COMPONENT_TO_KEY[label]
        value = ats_result.get(key, analysis_result.get(key, 0))
        score = max(0.0, min(100.0, _safe_float(value)))
        contribution = score * (weight / 100.0)
        rows.append(
            {
                "component": label,
                "score": round(score, 1),
                "weight": weight,
                "contribution": round(contribution, 1),
            }
        )
    return rows


def _sentence_split(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", _clean(text)).strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\n+|(?<=;)\s+", cleaned)
    return [s.strip(" -•\t") for s in sentences if len(s.strip()) >= 20]


def _terms(values: Sequence[str]) -> List[str]:
    terms: List[str] = []
    for value in values:
        value = _clean(value).lower()
        if not value:
            continue
        # Prefer multi-word phrases first, then useful tokens.
        if len(value.split()) > 1:
            terms.append(value)
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#./_-]{1,}", value):
            token = token.lower().strip("._/-")
            if len(token) >= 3 and token not in STOPWORDS:
                terms.append(token)
    return _unique(terms)


def _evidence_sentences(
    resume_text: str,
    targets: Sequence[str],
    limit: int = 6,
) -> List[str]:
    """Find resume sentences containing one or more known evidence terms."""
    sentences = _sentence_split(resume_text)
    target_terms = _terms(targets)
    if not sentences or not target_terms:
        return []

    ranked: List[Tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        normalized = sentence.lower()
        hits = 0
        for term in target_terms:
            if term in normalized:
                hits += 1
        if hits:
            ranked.append((hits, -index, sentence))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return _unique([item[2] for item in ranked])[:limit]


def _extract_lists(analysis_result: Dict[str, Any], ats_result: Dict[str, Any]) -> Dict[str, List[str]]:
    matched_skills = _unique(
        _flatten_values(ats_result.get("matched_skills"))
        + _flatten_values(analysis_result.get("matched_skills"))
    )
    missing_skills = _unique(
        _flatten_values(ats_result.get("missing_skills"))
        + _flatten_values(analysis_result.get("missing_skills"))
    )
    matched_keywords = _unique(
        _flatten_values(ats_result.get("matched_keywords"))
        + _flatten_values(analysis_result.get("matched_keywords"))
    )
    missing_keywords = _unique(
        _flatten_values(ats_result.get("missing_keywords"))
        + _flatten_values(analysis_result.get("missing_keywords"))
    )

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
    }


def _requirement_rows(candidate_matrix: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    return [dict(row) for row in (candidate_matrix or []) if isinstance(row, dict)]


def _requirement_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    required = [r for r in rows if _clean(r.get("category")).lower() == "required skill"]
    matched_statuses = {
        "matched",
        "detected in resume",
        "meets requirement",
        "detected",
    }
    missing_statuses = {"missing", "not detected"}

    matched = sum(1 for r in required if _clean(r.get("candidate_status")).lower() in matched_statuses)
    missing = sum(1 for r in required if _clean(r.get("candidate_status")).lower() in missing_statuses)
    return {
        "required_total": len(required),
        "required_matched": matched,
        "required_missing": missing,
    }


def build_explainability(
    *,
    resume_text: str,
    jd_text: str,
    analysis_result: Dict[str, Any] | None,
    ats_result: Dict[str, Any] | None,
    intelligence: Dict[str, Any] | None,
    quality: Dict[str, Any] | None,
    requirements: Dict[str, Any] | None,
    candidate_matrix: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a deterministic evidence package for one candidate."""
    analysis_result = analysis_result or {}
    ats_result = ats_result or {}
    intelligence = intelligence or {}
    quality = quality or {}
    requirements = requirements or {}
    rows = _requirement_rows(candidate_matrix)

    signals = _extract_lists(analysis_result, ats_result)
    breakdown = score_breakdown(ats_result, analysis_result)

    ats_score = max(
        0.0,
        min(
            100.0,
            _safe_float(
                ats_result.get("ats_score", analysis_result.get("ats_score", 0))
            ),
        ),
    )

    required = requirements.get("required_skills", {}) or {}
    preferred = requirements.get("preferred_skills", {}) or {}
    required_matched = _unique(required.get("matched", []))
    required_missing = _unique(required.get("missing", []))
    preferred_matched = _unique(preferred.get("matched", []))
    preferred_missing = _unique(preferred.get("missing", []))

    requirement_counts = _requirement_counts(rows)

    evidence_targets = _unique(
        signals["matched_skills"]
        + required_matched
        + signals["matched_keywords"]
    )
    resume_evidence = _evidence_sentences(
        resume_text,
        evidence_targets,
        limit=8,
    )

    missing_required = _unique(required_missing + signals["missing_skills"])
    verification_targets = _unique(
        required_missing
        + preferred_missing
        + signals["missing_keywords"]
    )

    semantic_score = max(0.0, min(100.0, _safe_float(ats_result.get("semantic_score", analysis_result.get("semantic_score", 0)))))
    lexical_score = max(0.0, min(100.0, _safe_float(ats_result.get("lexical_score", analysis_result.get("lexical_score", 0)))))
    keyword_score = max(0.0, min(100.0, _safe_float(ats_result.get("keyword_score", analysis_result.get("keyword_score", 0)))))
    skill_score = max(0.0, min(100.0, _safe_float(ats_result.get("skill_score", analysis_result.get("skill_score", 0)))))

    strengths = _unique(
        ats_result.get("strengths", [])
        + signals["matched_skills"]
        + required_matched
    )[:10]

    gaps = _unique(
        missing_required
        + verification_targets
    )[:10]

    return {
        "ats_score": round(ats_score, 1),
        "score_breakdown": breakdown,
        "signals": {
            "skill_score": round(skill_score, 1),
            "semantic_score": round(semantic_score, 1),
            "lexical_score": round(lexical_score, 1),
            "keyword_score": round(keyword_score, 1),
        },
        "matched_skills": signals["matched_skills"],
        "missing_skills": signals["missing_skills"],
        "matched_keywords": signals["matched_keywords"],
        "missing_keywords": signals["missing_keywords"],
        "required_matched": required_matched,
        "required_missing": required_missing,
        "preferred_matched": preferred_matched,
        "preferred_missing": preferred_missing,
        "requirement_counts": requirement_counts,
        "resume_evidence": resume_evidence,
        "strengths": strengths,
        "gaps": gaps,
        "verification_targets": verification_targets[:10],
        "resume_quality": round(_safe_float(quality.get("quality_score", 0)), 1),
        "experience_years": round(_safe_float(intelligence.get("experience_years", 0)), 1),
        "candidate_name": _clean(intelligence.get("name")) or "Candidate",
        "resume_name": _clean(intelligence.get("source_filename")),
        "jd_available": bool(_clean(jd_text)),
        "limitations": [
            "The ATS score uses the existing project-defined weights: Skill 35%, Semantic 30%, Lexical 20%, Keyword 15%.",
            "Semantic similarity is an aggregate signal; this view does not claim that every semantic similarity is a verified requirement match.",
            "Not detected in a resume is evidence absence, not proof that the candidate does not possess the skill.",
            "Resume evidence is extracted from the supplied text and is intended for recruiter verification.",
        ],
    }


def explanation_to_text(explanation: Dict[str, Any]) -> str:
    """Create a plain-text export for the explainability report."""
    lines = [
        "EXPLAINABLE MATCHING & EVIDENCE REPORT",
        "=" * 44,
        f"Candidate: {explanation.get('candidate_name', 'Candidate')}",
        f"Resume: {explanation.get('resume_name', '')}",
        f"Existing ATS Score: {explanation.get('ats_score', 0):.1f}%",
        "",
        "SCORE BREAKDOWN",
    ]

    for row in explanation.get("score_breakdown", []):
        lines.append(
            f"- {row['component']}: {row['score']:.1f}% "
            f"(weight {row['weight']:.0f}%, contribution {row['contribution']:.1f})"
        )

    counts = explanation.get("requirement_counts", {})
    lines.extend(
        [
            "",
            "REQUIREMENT COVERAGE",
            f"- Required skills matched: {', '.join(explanation.get('required_matched', [])) or 'None'}",
            f"- Required skills missing/not detected: {', '.join(explanation.get('required_missing', [])) or 'None'}",
            f"- Requirement matrix: {counts.get('required_matched', 0)} matched / {counts.get('required_total', 0)} required skill signals",
            f"- Preferred skills matched: {', '.join(explanation.get('preferred_matched', [])) or 'None'}",
            f"- Preferred skills missing/not detected: {', '.join(explanation.get('preferred_missing', [])) or 'None'}",
            "",
            "MATCHED SKILLS",
            ", ".join(explanation.get("matched_skills", [])) or "None",
            "",
            "MATCHED KEYWORDS",
            ", ".join(explanation.get("matched_keywords", [])) or "None",
            "",
            "RESUME EVIDENCE",
        ]
    )

    evidence = explanation.get("resume_evidence", [])
    if evidence:
        for index, sentence in enumerate(evidence, start=1):
            lines.append(f"{index}. {sentence}")
    else:
        lines.append("No matching evidence sentence was extracted from the supplied resume text.")

    lines.extend(["", "VERIFICATION AREAS"])
    verification = explanation.get("verification_targets", [])
    lines.append(", ".join(verification) or "No specific verification targets generated.")

    lines.extend(["", "LIMITATIONS"])
    for item in explanation.get("limitations", []):
        lines.append(f"- {item}")

    return "\n".join(lines)
