
"""
Hybrid ATS scoring engine.

The score combines:

    Skill Match       35%
    Semantic Match    30%
    Lexical Match     20%
    Keyword Match     15%

These weights are project-defined and are not an official
formula used by any particular ATS vendor.
"""

from typing import List, Dict


# ============================================================
# WEIGHTS
# ============================================================

SKILL_WEIGHT = 0.35
SEMANTIC_WEIGHT = 0.30
LEXICAL_WEIGHT = 0.20
KEYWORD_WEIGHT = 0.15


# ============================================================
# SAFE SCORE
# ============================================================

def safe_score(
    value: float
) -> float:

    return max(
        0.0,
        min(
            100.0,
            float(value)
        )
    )


# ============================================================
# COMPONENT SCORE
# ============================================================

def calculate_coverage_score(
    matched: List[str],
    missing: List[str]
) -> float:

    total = (
        len(matched)
        + len(missing)
    )

    if total == 0:
        return 0.0

    return round(
        (
            len(matched)
            / total
        ) * 100,
        2
    )


# ============================================================
# FINAL ATS SCORE
# ============================================================

def calculate_ats_score(
    skill_score: float,
    semantic_score: float,
    lexical_score: float,
    keyword_score: float
) -> float:

    skill_score = safe_score(
        skill_score
    )

    semantic_score = safe_score(
        semantic_score
    )

    lexical_score = safe_score(
        lexical_score
    )

    keyword_score = safe_score(
        keyword_score
    )

    score = (

        skill_score
        * SKILL_WEIGHT

        +

        semantic_score
        * SEMANTIC_WEIGHT

        +

        lexical_score
        * LEXICAL_WEIGHT

        +

        keyword_score
        * KEYWORD_WEIGHT
    )

    return round(
        safe_score(score),
        2
    )


# ============================================================
# STATUS
# ============================================================

def get_score_status(
    score: float
) -> str:

    score = safe_score(
        score
    )

    if score >= 80:
        return "Strong Match"

    if score >= 60:
        return "Moderate Match"

    return "Low Match"


# ============================================================
# SCORE BREAKDOWN
# ============================================================

def get_score_breakdown(
    skill_score: float,
    semantic_score: float,
    lexical_score: float,
    keyword_score: float
) -> Dict:

    return {

        "Skill Match":
            round(
                safe_score(skill_score)
                * SKILL_WEIGHT,
                2
            ),

        "Semantic Match":
            round(
                safe_score(semantic_score)
                * SEMANTIC_WEIGHT,
                2
            ),

        "Lexical Match":
            round(
                safe_score(lexical_score)
                * LEXICAL_WEIGHT,
                2
            ),

        "Keyword Match":
            round(
                safe_score(keyword_score)
                * KEYWORD_WEIGHT,
                2
            )
    }


# ============================================================
# STRENGTHS
# ============================================================

def generate_strengths(
    skill_score: float,
    semantic_score: float,
    lexical_score: float,
    keyword_score: float,
    matched_skills: List[str]
) -> List[str]:

    strengths = []

    if skill_score >= 80:

        strengths.append(
            "Strong coverage of required technical skills."
        )

    elif skill_score >= 60:

        strengths.append(
            "Good coverage of required technical skills."
        )

    if semantic_score >= 80:

        strengths.append(
            "Strong semantic alignment with the role."
        )

    elif semantic_score >= 60:

        strengths.append(
            "Good semantic alignment with the role."
        )

    if lexical_score >= 75:

        strengths.append(
            "Strong textual alignment with the Job Description."
        )

    if keyword_score >= 80:

        strengths.append(
            "Strong coverage of important Job Description keywords."
        )

    if matched_skills:

        preview = ", ".join(
            matched_skills[:6]
        )

        strengths.append(
            f"Relevant skills detected: {preview}."
        )

    if not strengths:

        strengths.append(
            "Some relevant information was detected, "
            "but alignment is limited."
        )

    return strengths


# ============================================================
# SKILL GAPS
# ============================================================

def generate_skill_gaps(
    missing_skills: List[str]
) -> List[str]:

    if not missing_skills:

        return [
            "No missing skills detected "
            "from the current skill database."
        ]

    return [
        f"{skill} was not detected in the resume."
        for skill in missing_skills
    ]


# ============================================================
# IMPROVEMENT SUGGESTIONS
# ============================================================

def generate_improvement_suggestions(
    skill_score: float,
    semantic_score: float,
    lexical_score: float,
    keyword_score: float,
    missing_skills: List[str],
    missing_keywords: List[str]
) -> List[str]:

    suggestions = []

    if missing_skills:

        preview = ", ".join(
            missing_skills[:6]
        )

        suggestions.append(
            f"If you genuinely possess them, consider "
            f"adding evidence for: {preview}."
        )

    if missing_keywords:

        preview = ", ".join(
            missing_keywords[:6]
        )

        suggestions.append(
            f"Consider naturally including relevant JD "
            f"terms such as: {preview}."
        )

    if semantic_score < 60:

        suggestions.append(
            "Review whether the resume clearly describes "
            "experience relevant to the role."
        )

    if lexical_score < 50:

        suggestions.append(
            "Align the wording of relevant experience, "
            "projects and responsibilities more closely "
            "with the role."
        )

    if skill_score < 60:

        suggestions.append(
            "Highlight relevant projects, coursework, "
            "certifications or experience demonstrating "
            "required skills."
        )

    if keyword_score < 60:

        suggestions.append(
            "Improve relevant keyword coverage without "
            "adding skills or experience you do not have."
        )

    if not suggestions:

        suggestions.append(
            "The resume shows good alignment. Continue "
            "using specific, truthful and measurable "
            "achievement statements."
        )

    return suggestions


# ============================================================
# COMPLETE ATS ANALYSIS
# ============================================================

def analyze_ats(
    skill_score: float,
    semantic_score: float,
    lexical_score: float,
    matched_keywords: List[str],
    missing_keywords: List[str],
    matched_skills: List[str],
    missing_skills: List[str]
) -> Dict:

    keyword_score = calculate_coverage_score(
        matched_keywords,
        missing_keywords
    )

    ats_score = calculate_ats_score(
        skill_score=skill_score,
        semantic_score=semantic_score,
        lexical_score=lexical_score,
        keyword_score=keyword_score
    )

    status = get_score_status(
        ats_score
    )

    breakdown = get_score_breakdown(
        skill_score,
        semantic_score,
        lexical_score,
        keyword_score
    )

    strengths = generate_strengths(
        skill_score,
        semantic_score,
        lexical_score,
        keyword_score,
        matched_skills
    )

    skill_gaps = generate_skill_gaps(
        missing_skills
    )

    suggestions = generate_improvement_suggestions(
        skill_score,
        semantic_score,
        lexical_score,
        keyword_score,
        missing_skills,
        missing_keywords
    )

    return {

        "ats_score":
            ats_score,

        "status":
            status,

        "skill_score":
            round(
                skill_score,
                2
            ),

        "semantic_score":
            round(
                semantic_score,
                2
            ),

        "lexical_score":
            round(
                lexical_score,
                2
            ),

        "keyword_score":
            round(
                keyword_score,
                2
            ),

        "breakdown":
            breakdown,

        "strengths":
            strengths,

        "skill_gaps":
            skill_gaps,

        "suggestions":
            suggestions
    }