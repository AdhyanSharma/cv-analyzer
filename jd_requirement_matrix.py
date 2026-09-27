
"""
JD Requirement Matrix V7

Additive recruiter-facing layer for the existing JD Intelligence engine.

It converts deterministic JD analysis + optional RAG/Gemini enrichment
into a requirement matrix and, when candidate data is supplied, adds
descriptive candidate coverage.

It intentionally does NOT change ATS weights or make hiring decisions.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence


CATEGORY_LABELS = {
    "required_skill": "Required Skill",
    "preferred_skill": "Preferred Skill",
    "experience": "Experience",
    "education": "Education",
    "responsibility": "Responsibility",
    "soft_skill": "Soft Skill",
    "domain": "Domain",
    "concept": "Concept",
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(x) for x in value]
    except TypeError:
        return [str(value)]


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        item = _clean(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _evidence_map(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    output = {}
    for key, values in value.items():
        evidence = _unique(_list(values))
        if evidence:
            output[_clean(key).lower()] = evidence
    return output


def _find_evidence(
    name: str,
    evidence_map: Dict[str, List[str]],
    fallback_lines: Sequence[str] = (),
) -> List[str]:
    key = _clean(name).lower()
    if key in evidence_map:
        return evidence_map[key]

    normalized = re.sub(r"[^a-z0-9]+", " ", key).strip()
    for evidence_key, values in evidence_map.items():
        normalized_key = re.sub(
            r"[^a-z0-9]+",
            " ",
            evidence_key,
        ).strip()
        if normalized == normalized_key or normalized in normalized_key or normalized_key in normalized:
            return values

    return _unique(
        [
            line
            for line in fallback_lines
            if key and key in _clean(line).lower()
        ]
    )[:3]


def _add_row(
    rows: List[Dict[str, Any]],
    *,
    name: str,
    category: str,
    priority: str,
    source: str,
    evidence: Optional[Sequence[str]] = None,
    section: str = "",
) -> None:
    name = _clean(name)
    if not name:
        return

    rows.append(
        {
            "requirement": name,
            "category": CATEGORY_LABELS.get(
                category,
                category.replace("_", " ").title(),
            ),
            "priority": priority,
            "source": source,
            "section": _clean(section),
            "evidence": _unique(evidence or []),
        }
    )


def build_requirement_matrix(
    jd_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    analysis = jd_analysis or {}
    rows: List[Dict[str, Any]] = []

    sections = analysis.get("sections", {}) or {}
    required_lines = _list(sections.get("required", []))
    preferred_lines = _list(sections.get("preferred", []))

    required_evidence = _evidence_map(
        analysis.get("required_evidence", {})
    )
    preferred_evidence = _evidence_map(
        analysis.get("preferred_evidence", {})
    )

    for skill in _unique(
        _list(analysis.get("required_skills", []))
    ):
        _add_row(
            rows,
            name=skill,
            category="required_skill",
            priority="Must Have",
            source="Section-Aware NLP",
            evidence=_find_evidence(
                skill,
                required_evidence,
                required_lines,
            ),
            section="Required",
        )

    for skill in _unique(
        _list(analysis.get("preferred_skills", []))
    ):
        _add_row(
            rows,
            name=skill,
            category="preferred_skill",
            priority="Preferred",
            source="Section-Aware NLP",
            evidence=_find_evidence(
                skill,
                preferred_evidence,
                preferred_lines,
            ),
            section="Preferred",
        )

    experience = analysis.get("experience", {}) or {}
    years = experience.get("required_years")
    raw_matches = _unique(
        _list(experience.get("raw_matches", []))
    )
    if (isinstance(years, (int, float)) and years > 0) or raw_matches:
        label = (
            f"{years:g}+ years experience"
            if isinstance(years, (int, float)) and years > 0
            else raw_matches[0]
        )
        _add_row(
            rows,
            name=label,
            category="experience",
            priority="Must Have",
            source="Requirement Extraction",
            evidence=raw_matches,
            section="Experience / Requirements",
        )

    for item in _unique(
        _list(analysis.get("education", []))
    )[:8]:
        _add_row(
            rows,
            name=item,
            category="education",
            priority="Qualification",
            source="JD Section Parsing",
            evidence=[item],
            section="Education",
        )

    for item in _unique(
        _list(analysis.get("responsibilities", []))
    )[:10]:
        _add_row(
            rows,
            name=item,
            category="responsibility",
            priority="Context",
            source="JD Section Parsing",
            evidence=[item],
            section="Responsibilities",
        )

    # Support both V4/V5/V6 keys.
    gemini = (
        analysis.get("rag_gemini")
        or analysis.get("gemini")
        or {}
    )

    if isinstance(gemini, dict):
        for key, category, priority in [
            ("domain", "domain", "Context"),
            ("domain_keywords", "domain", "Context"),
            ("concept_keywords", "concept", "Context"),
            ("soft_skills", "soft_skill", "Supporting"),
            ("qualification_keywords", "education", "Supporting"),
        ]:
            for item in _unique(
                _list(gemini.get(key, []))
            )[:10]:
                _add_row(
                    rows,
                    name=item,
                    category=category,
                    priority=priority,
                    source="RAG + Gemini",
                    section="AI Enrichment",
                )

        for item in _unique(
            _list(
                gemini.get(
                    "responsibility_themes",
                    [],
                )
            )
        )[:10]:
            _add_row(
                rows,
                name=item,
                category="responsibility",
                priority="Context",
                source="RAG + Gemini",
                section="AI Enrichment",
            )

    unique = []
    seen = set()
    for row in rows:
        key = (
            row["requirement"].lower(),
            row["category"].lower(),
            row["priority"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    return unique


def requirements_to_dataframe(
    matrix: Sequence[Dict[str, Any]],
):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "Requirement": row.get("requirement", ""),
                "Category": row.get("category", ""),
                "Priority": row.get("priority", ""),
                "Source": row.get("source", ""),
                "Section": row.get("section", ""),
                "Evidence": " | ".join(
                    _unique(
                        _list(row.get("evidence", []))
                    )
                ),
            }
            for row in matrix
        ]
    )


def _candidate_skill_set(
    intelligence: Dict[str, Any],
) -> set[str]:
    intelligence = intelligence or {}
    skills = intelligence.get("all_skills", [])
    if not skills:
        skills = intelligence.get("skills", [])

    values = []
    if isinstance(skills, dict):
        for group in skills.values():
            values.extend(_list(group))
    else:
        values.extend(_list(skills))

    return {
        _clean(x).lower()
        for x in values
        if _clean(x)
    }


def _skill_seen(
    requirement: str,
    skills: set[str],
) -> bool:
    target = _clean(requirement).lower()
    if not target:
        return False
    if target in skills:
        return True

    target_tokens = set(
        re.findall(
            r"[a-z0-9+#.]+",
            target,
        )
    )

    for skill in skills:
        if target in skill or skill in target:
            return True
        skill_tokens = set(
            re.findall(
                r"[a-z0-9+#.]+",
                skill,
            )
        )
        if target_tokens and target_tokens <= skill_tokens:
            return True

    return False


def build_candidate_requirement_matrix(
    *,
    jd_analysis: Dict[str, Any],
    candidate_requirements: Optional[Dict[str, Any]] = None,
    candidate_intelligence: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    matrix = build_requirement_matrix(
        jd_analysis
    )

    req = candidate_requirements or {}
    intelligence = candidate_intelligence or {}

    required_data = req.get("required_skills", {}) or {}
    preferred_data = req.get("preferred_skills", {}) or {}

    matched_required = {
        _clean(x).lower()
        for x in _list(required_data.get("matched", []))
    }
    missing_required = {
        _clean(x).lower()
        for x in _list(required_data.get("missing", []))
    }
    matched_preferred = {
        _clean(x).lower()
        for x in _list(preferred_data.get("matched", []))
    }
    missing_preferred = {
        _clean(x).lower()
        for x in _list(preferred_data.get("missing", []))
    }

    candidate_skills = _candidate_skill_set(
        intelligence
    )

    output = []

    for row in matrix:
        item = dict(row)
        requirement = _clean(
            row.get("requirement", "")
        )
        normalized = requirement.lower()
        category = row.get("category", "")

        status = "Context"
        source = "JD context only"

        if category == "Required Skill":
            if normalized in matched_required:
                status = "Matched"
                source = "Requirement matching"
            elif normalized in missing_required:
                status = "Missing"
                source = "Requirement matching"
            elif _skill_seen(
                requirement,
                candidate_skills,
            ):
                status = "Detected in Resume"
                source = "Resume skill extraction"
            else:
                status = "Not detected"
                source = "Resume skill extraction"

        elif category == "Preferred Skill":
            if normalized in matched_preferred:
                status = "Matched"
                source = "Requirement matching"
            elif normalized in missing_preferred:
                status = "Missing"
                source = "Requirement matching"
            elif _skill_seen(
                requirement,
                candidate_skills,
            ):
                status = "Detected in Resume"
                source = "Resume skill extraction"
            else:
                status = "Not detected"
                source = "Resume skill extraction"

        elif category == "Experience":
            info = req.get("experience", {}) or {}
            status = str(
                info.get(
                    "status",
                    "Not assessed",
                )
            )
            source = "Experience requirement matching"

        elif category == "Education":
            info = req.get("education", {}) or {}
            status = str(
                info.get(
                    "status",
                    "Not assessed",
                )
            )
            source = "Education requirement matching"

        item["candidate_status"] = status
        item["candidate_match_source"] = source
        output.append(item)

    return output


def build_requirement_summary(
    matrix: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    summary = {
        "total": len(list(matrix)),
        "required": 0,
        "preferred": 0,
        "experience": 0,
        "education": 0,
        "responsibilities": 0,
        "soft_skills": 0,
        "domain": 0,
        "concepts": 0,
    }

    for row in matrix:
        category = str(
            row.get("category", "")
        ).lower()

        mapping = {
            "required skill": "required",
            "preferred skill": "preferred",
            "experience": "experience",
            "education": "education",
            "responsibility": "responsibilities",
            "soft skill": "soft_skills",
            "domain": "domain",
            "concept": "concepts",
        }

        key = mapping.get(category)
        if key:
            summary[key] += 1

    return summary


def requirement_matrix_to_text(
    matrix: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "JD REQUIREMENT MATRIX",
        "=" * 22,
        "",
    ]

    for index, row in enumerate(
        matrix,
        start=1,
    ):
        lines.extend(
            [
                f"{index}. {row.get('requirement', '')}",
                f"   Category: {row.get('category', '')}",
                f"   Priority: {row.get('priority', '')}",
                f"   Source: {row.get('source', '')}",
                f"   Section: {row.get('section', '')}",
            ]
        )

        evidence = _unique(
            _list(row.get("evidence", []))
        )

        for value in evidence:
            lines.append(
                f"   Evidence: {value}"
            )

        if "candidate_status" in row:
            lines.append(
                "   Candidate Status: "
                + str(row.get("candidate_status", ""))
            )

        lines.append("")

    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    sample = {
        "required_skills": ["Python", "SQL", "Oracle"],
        "preferred_skills": ["Java", "AWS"],
        "required_evidence": {
            "Python": ["Required Skills: Python, SQL, Oracle."]
        },
        "experience": {
            "required_years": 2,
            "raw_matches": ["2+ years of experience"],
        },
        "education": ["Bachelor degree in Computer Science."],
        "responsibilities": [
            "Build and maintain web applications."
        ],
        "gemini": {
            "domain": ["Web Development"],
            "soft_skills": ["Team collaboration"],
            "concept_keywords": ["Backend Development"],
        },
    }

    matrix = build_requirement_matrix(sample)
    candidate_matrix = build_candidate_requirement_matrix(
        jd_analysis=sample,
        candidate_requirements={
            "required_skills": {
                "matched": ["Python", "SQL"],
                "missing": ["Oracle"],
            },
            "preferred_skills": {
                "matched": ["Java"],
                "missing": ["AWS"],
            },
            "experience": {
                "status": "Meets requirement"
            },
            "education": {
                "status": "Detected"
            },
        },
        candidate_intelligence={
            "all_skills": [
                "Python",
                "SQL",
                "Java",
            ]
        },
    )

    assert matrix
    assert any(
        row["candidate_status"] == "Matched"
        for row in candidate_matrix
        if row["category"] == "Required Skill"
    )
    assert any(
        row["candidate_status"] == "Missing"
        for row in candidate_matrix
        if row["category"] == "Required Skill"
    )

    print("JD Requirement Matrix V7: PASSED")
    print("Rows:", len(matrix))
    print(
        json.dumps(
            build_requirement_summary(matrix),
            indent=2,
        )
    )
