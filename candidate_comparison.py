"""
Candidate Comparison Engine
---------------------------

Deterministic side-by-side comparison for two analyzed candidates.

This module does not declare a winner or ranking. It presents the
same project-defined analytical signals for both candidates and
shows overlaps / differences that a recruiter can inspect.

It can also build a comparison evidence payload for the RAG copilot.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


COMPARISON_METRICS = [
    ("ATS Score", "ATS Score"),
    ("Requirement Match", "Requirement Match"),
    ("Resume Quality", "Resume Quality"),
    ("Semantic Score", "Semantic Score"),
    ("Skill Score", "Skill Score"),
    ("Lexical Score", "Lexical Score"),
    ("Keyword Score", "Keyword Score"),
    ("Experience Years", "Experience Years"),
    ("Skills Detected", "Skills Detected"),
    ("Required Skills Matched", "Required Skills Matched"),
    ("Required Skills Missing", "Required Skills Missing"),
    ("Preferred Skills Matched", "Preferred Skills Matched"),
    ("Section Completeness", "Section Completeness"),
    ("Contact Completeness", "Contact Completeness"),
]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, Iterable):
        result = []
        for item in value:
            text = str(item or "").strip()
            if text:
                result.append(text)
        return result

    text = str(value).strip()
    return [text] if text else []


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []

    for value in values:
        item = str(value).strip()
        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def _case_map(values: Iterable[str]) -> Dict[str, str]:
    return {
        str(value).strip().lower(): str(value).strip()
        for value in values
        if str(value).strip()
    }


def extract_candidate_skills(
    candidate_cache: Dict[str, Any],
) -> List[str]:
    intelligence = (
        candidate_cache.get(
            "intelligence",
            {},
        )
        or {}
    )

    skills = intelligence.get(
        "all_skills",
        [],
    )

    return _unique(_list(skills))


def extract_required_skills(
    candidate_cache: Dict[str, Any],
) -> Dict[str, List[str]]:
    requirements = (
        candidate_cache.get(
            "requirements",
            {},
        )
        or {}
    )

    required = (
        requirements.get(
            "required_skills",
            {},
        )
        or {}
    )

    return {
        "matched": _unique(
            _list(required.get("matched", []))
        ),
        "missing": _unique(
            _list(required.get("missing", []))
        ),
    }


def extract_preferred_skills(
    candidate_cache: Dict[str, Any],
) -> Dict[str, List[str]]:
    requirements = (
        candidate_cache.get(
            "requirements",
            {},
        )
        or {}
    )

    preferred = (
        requirements.get(
            "preferred_skills",
            {},
        )
        or {}
    )

    return {
        "matched": _unique(
            _list(preferred.get("matched", []))
        ),
        "missing": _unique(
            _list(preferred.get("missing", []))
        ),
    }


def compare_metric_values(
    value_a: Any,
    value_b: Any,
    precision: int = 1,
) -> Dict[str, Any]:
    """
    Compare two numeric values descriptively.
    """
    a = _number(value_a)
    b = _number(value_b)

    return {
        "candidate_a": round(a, precision),
        "candidate_b": round(b, precision),
        "difference_a_minus_b": round(
            a - b,
            precision,
        ),
        "absolute_difference": round(
            abs(a - b),
            precision,
        ),
    }


def build_candidate_comparison(
    *,
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    cache_a: Dict[str, Any],
    cache_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a complete side-by-side comparison object.
    """

    name_a = _clean(
        row_a.get("Candidate")
    ) or "Candidate A"

    name_b = _clean(
        row_b.get("Candidate")
    ) or "Candidate B"

    metrics = {}

    for label, key in COMPARISON_METRICS:
        precision = 1

        if key == "Experience Years":
            precision = 1
        elif key in {
            "Skills Detected",
            "Required Skills Matched",
            "Required Skills Missing",
            "Preferred Skills Matched",
        }:
            precision = 0

        metrics[label] = compare_metric_values(
            row_a.get(key),
            row_b.get(key),
            precision=precision,
        )

    skills_a = extract_candidate_skills(
        cache_a
    )

    skills_b = extract_candidate_skills(
        cache_b
    )

    map_a = _case_map(skills_a)
    map_b = _case_map(skills_b)

    overlap_keys = set(
        map_a
    ) & set(
        map_b
    )

    unique_a_keys = set(
        map_a
    ) - set(
        map_b
    )

    unique_b_keys = set(
        map_b
    ) - set(
        map_a
    )

    required_a = extract_required_skills(
        cache_a
    )

    required_b = extract_required_skills(
        cache_b
    )

    preferred_a = extract_preferred_skills(
        cache_a
    )

    preferred_b = extract_preferred_skills(
        cache_b
    )

    return {
        "candidate_a": {
            "name": name_a,
            "resume": _clean(
                row_a.get("Resume")
            ),
        },
        "candidate_b": {
            "name": name_b,
            "resume": _clean(
                row_b.get("Resume")
            ),
        },
        "metrics": metrics,
        "skills": {
            "shared": sorted(
                map_a[key]
                for key in overlap_keys
            ),
            "only_a": sorted(
                map_a[key]
                for key in unique_a_keys
            ),
            "only_b": sorted(
                map_b[key]
                for key in unique_b_keys
            ),
        },
        "required_skills": {
            "a": required_a,
            "b": required_b,
        },
        "preferred_skills": {
            "a": preferred_a,
            "b": preferred_b,
        },
    }


def comparison_to_text(
    comparison: Dict[str, Any]
) -> str:
    """
    Export a descriptive comparison.
    """

    a = comparison.get(
        "candidate_a",
        {},
    )

    b = comparison.get(
        "candidate_b",
        {},
    )

    lines = [
        "CANDIDATE COMPARISON",
        "=" * 60,
        f"Candidate A: {a.get('name', 'Candidate A')}",
        f"Resume: {a.get('resume', '')}",
        "",
        f"Candidate B: {b.get('name', 'Candidate B')}",
        f"Resume: {b.get('resume', '')}",
        "",
        "METRICS",
        "-" * 30,
    ]

    for label, values in (
        comparison.get(
            "metrics",
            {},
        )
        or {}
    ).items():
        lines.append(
            f"{label}: "
            f"A={values.get('candidate_a')} | "
            f"B={values.get('candidate_b')} | "
            f"Difference(A-B)={values.get('difference_a_minus_b')}"
        )

    skills = comparison.get(
        "skills",
        {},
    ) or {}

    lines.extend(
        [
            "",
            "SKILL OVERLAP",
            "-" * 30,
            "Shared: "
            + ", ".join(
                skills.get(
                    "shared",
                    [],
                )
            ),
            (
                f"Only A: "
                + ", ".join(
                    skills.get(
                        "only_a",
                        [],
                    )
                )
            ),
            (
                f"Only B: "
                + ", ".join(
                    skills.get(
                        "only_b",
                        [],
                    )
                )
            ),
        ]
    )

    for title, data in [
        (
            "REQUIRED SKILLS - CANDIDATE A",
            (
                comparison.get(
                    "required_skills",
                    {},
                )
                or {}
            ).get("a", {}),
        ),
        (
            "REQUIRED SKILLS - CANDIDATE B",
            (
                comparison.get(
                    "required_skills",
                    {},
                )
                or {}
            ).get("b", {}),
        ),
        (
            "PREFERRED SKILLS - CANDIDATE A",
            (
                comparison.get(
                    "preferred_skills",
                    {},
                )
                or {}
            ).get("a", {}),
        ),
        (
            "PREFERRED SKILLS - CANDIDATE B",
            (
                comparison.get(
                    "preferred_skills",
                    {},
                )
                or {}
            ).get("b", {}),
        ),
    ]:
        lines.extend(
            [
                "",
                title,
                "-" * 30,
                "Matched: "
                + ", ".join(
                    data.get(
                        "matched",
                        [],
                    )
                ),
                "Missing: "
                + ", ".join(
                    data.get(
                        "missing",
                        [],
                    )
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Note: This is a descriptive comparison of project-defined "
            "matching signals and extracted evidence. It does not declare "
            "a hiring decision or winner.",
        ]
    )

    return "\n".join(lines)


def build_comparison_evidence(
    *,
    comparison: Dict[str, Any],
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    cache_a: Dict[str, Any],
    cache_b: Dict[str, Any],
) -> str:
    """
    Create evidence text for an optional Gemini comparison answer.
    """

    def summarize(
        label: str,
        row: Dict[str, Any],
        cache: Dict[str, Any],
    ) -> List[str]:
        intelligence = (
            cache.get(
                "intelligence",
                {},
            )
            or {}
        )

        analysis = (
            cache.get(
                "analysis",
                {},
            )
            or {}
        )

        requirements = (
            cache.get(
                "requirements",
                {},
            )
            or {}
        )

        required = (
            requirements.get(
                "required_skills",
                {},
            )
            or {}
        )

        preferred = (
            requirements.get(
                "preferred_skills",
                {},
            )
            or {}
        )

        return [
            f"{label} name: {row.get('Candidate', '')}",
            f"{label} resume: {row.get('Resume', '')}",
            f"{label} ATS Score: {row.get('ATS Score', '')}",
            f"{label} Requirement Match: {row.get('Requirement Match', '')}",
            f"{label} Resume Quality: {row.get('Resume Quality', '')}",
            f"{label} Semantic Score: {row.get('Semantic Score', '')}",
            f"{label} Skill Score: {row.get('Skill Score', '')}",
            f"{label} Experience Years: {row.get('Experience Years', '')}",
            f"{label} Detected Skills: {', '.join(extract_candidate_skills(cache))}",
            f"{label} Matched Skills: {', '.join(_list(analysis.get('matched_skills', [])))}",
            f"{label} Missing Skills: {', '.join(_list(analysis.get('missing_skills', [])))}",
            f"{label} Required Skills Matched: {', '.join(_list(required.get('matched', [])))}",
            f"{label} Required Skills Missing: {', '.join(_list(required.get('missing', [])))}",
            f"{label} Preferred Skills Matched: {', '.join(_list(preferred.get('matched', [])))}",
            f"{label} Preferred Skills Missing: {', '.join(_list(preferred.get('missing', [])))}",
            f"{label} Education: {intelligence.get('education', '')}",
            f"{label} Projects: {intelligence.get('projects', '')}",
            f"{label} Certifications: {intelligence.get('certifications', '')}",
        ]

    lines = summarize(
        "Candidate A",
        row_a,
        cache_a,
    )

    lines.extend(
        [
            "",
            "COMPARISON",
            "Candidate A name: "
            + comparison.get(
                "candidate_a",
                {},
            ).get("name", ""),
            "Candidate B name: "
            + comparison.get(
                "candidate_b",
                {},
            ).get("name", ""),
            "Shared skills: "
            + ", ".join(
                (
                    comparison.get(
                        "skills",
                        {},
                    )
                    or {}
                ).get(
                    "shared",
                    [],
                )
            ),
            "Skills only in Candidate A: "
            + ", ".join(
                (
                    comparison.get(
                        "skills",
                        {},
                    )
                    or {}
                ).get(
                    "only_a",
                    [],
                )
            ),
            "Skills only in Candidate B: "
            + ", ".join(
                (
                    comparison.get(
                        "skills",
                        {},
                    )
                    or {}
                ).get(
                    "only_b",
                    [],
                )
            ),
        ]
    )

    lines.extend(
        [
            "",
            *summarize(
                "Candidate B",
                row_b,
                cache_b,
            ),
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    row_a = {
        "Candidate": "Adhyan Sharma",
        "Resume": "Adhyan.pdf",
        "ATS Score": 82,
        "Requirement Match": 78,
        "Resume Quality": 90,
        "Semantic Score": 80,
        "Skill Score": 84,
        "Lexical Score": 74,
        "Keyword Score": 79,
        "Experience Years": 1.5,
        "Skills Detected": 5,
        "Required Skills Matched": 4,
        "Required Skills Missing": 1,
        "Preferred Skills Matched": 2,
        "Section Completeness": 92,
        "Contact Completeness": 100,
    }

    row_b = {
        "Candidate": "Test Candidate",
        "Resume": "Test.pdf",
        "ATS Score": 76,
        "Requirement Match": 81,
        "Resume Quality": 85,
        "Semantic Score": 74,
        "Skill Score": 79,
        "Lexical Score": 70,
        "Keyword Score": 72,
        "Experience Years": 2,
        "Skills Detected": 6,
        "Required Skills Matched": 4,
        "Required Skills Missing": 1,
        "Preferred Skills Matched": 1,
        "Section Completeness": 88,
        "Contact Completeness": 100,
    }

    cache_a = {
        "intelligence": {
            "all_skills": [
                "Python",
                "Machine Learning",
                "SQL",
            ],
            "education": "B.Tech AI",
            "projects": "CV Analyzer",
        },
        "analysis": {
            "matched_skills": [
                "Python",
                "Machine Learning",
            ],
            "missing_skills": [
                "Docker",
            ],
        },
        "requirements": {
            "required_skills": {
                "matched": [
                    "Python",
                ],
                "missing": [
                    "Docker",
                ],
            },
            "preferred_skills": {
                "matched": [
                    "SQL",
                ],
                "missing": [],
            },
        },
    }

    cache_b = {
        "intelligence": {
            "all_skills": [
                "Python",
                "SQL",
                "Docker",
            ],
            "education": "B.Tech CSE",
            "projects": "ML Project",
        },
        "analysis": {
            "matched_skills": [
                "Python",
                "SQL",
            ],
            "missing_skills": [
                "AWS",
            ],
        },
        "requirements": {
            "required_skills": {
                "matched": [
                    "Python",
                ],
                "missing": [
                    "AWS",
                ],
            },
            "preferred_skills": {
                "matched": [
                    "Docker",
                ],
                "missing": [],
            },
        },
    }

    comparison = build_candidate_comparison(
        row_a=row_a,
        row_b=row_b,
        cache_a=cache_a,
        cache_b=cache_b,
    )

    assert comparison["candidate_a"]["name"] == "Adhyan Sharma"
    assert "Python" in comparison["skills"]["shared"]
    assert "Machine Learning" in comparison["skills"]["only_a"]
    assert "Docker" in comparison["skills"]["only_b"]

    print(
        "Candidate comparison engine is working correctly."
    )
