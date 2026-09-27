"""Candidate 360-degree profile helpers."""

from typing import Any, Dict, Iterable, List


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def join_values(values: Any, fallback: str = "Not detected") -> str:
    if values is None:
        return fallback

    if isinstance(values, str):
        text = values.strip()
        return text or fallback

    if isinstance(values, Iterable):
        cleaned = [str(item).strip() for item in values if str(item).strip()]
        return ", ".join(cleaned) if cleaned else fallback

    return str(values)


def format_metric(value: Any, suffix: str = "%", decimals: int = 1) -> str:
    return f"{safe_float(value):.{decimals}f}{suffix}"


def get_contact_fields(intelligence: Dict) -> Dict[str, str]:
    return {
        "Email": str(intelligence.get("email") or ""),
        "Phone": str(intelligence.get("phone") or ""),
        "LinkedIn": str(intelligence.get("linkedin") or ""),
        "GitHub": str(intelligence.get("github") or ""),
    }


def build_candidate_snapshot(
    candidate_row: Dict,
    intelligence: Dict,
    ats: Dict,
    quality: Dict,
    requirements: Dict,
) -> Dict[str, Any]:
    experience = requirements.get("experience", {}) or {}

    return {
        "candidate": candidate_row.get("Candidate", "Candidate"),
        "resume": candidate_row.get("Resume", ""),
        "ats_score": safe_float(ats.get("ats_score", 0)),
        "quality_score": safe_float(quality.get("quality_score", 0)),
        "requirement_score": requirements.get("requirement_match_score"),
        "semantic_score": safe_float(ats.get("semantic_score", 0)),
        "lexical_score": safe_float(ats.get("lexical_score", 0)),
        "skill_score": safe_float(ats.get("skill_score", 0)),
        "keyword_score": safe_float(ats.get("keyword_score", 0)),
        "experience_years": safe_float(
            intelligence.get("experience_years", experience.get("candidate_years", 0))
        ),
        "skills_detected": len(intelligence.get("all_skills", []) or []),
        "education": intelligence.get("education", ""),
        "projects": intelligence.get("projects", ""),
        "certifications": intelligence.get("certifications", ""),
    }


def build_screening_summary(snapshot: Dict[str, Any]) -> str:
    ats = safe_float(snapshot.get("ats_score"))
    requirement = snapshot.get("requirement_score")
    quality = safe_float(snapshot.get("quality_score"))

    lines: List[str] = []
    lines.append(
        f"ATS match is {ats:.1f}%. Resume quality is {quality:.1f}%."
    )

    if requirement is not None:
        lines.append(
            f"Requirement coverage is {safe_float(requirement):.1f}% based on the extracted role requirements."
        )
    else:
        lines.append("Requirement coverage was not available for this candidate.")

    lines.append(
        f"Detected {safe_int(snapshot.get('skills_detected'))} technical skills and "
        f"approximately {safe_float(snapshot.get('experience_years')):.1f} years of experience."
    )

    return " ".join(lines)
