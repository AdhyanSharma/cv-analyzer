"""
AI Resume Screening Platform
============================

Candidate Requirement Matching Engine

Purpose:
    Compare structured candidate information against
    structured Job Description requirements.

Analyzes:
    - Required skill coverage
    - Preferred skill coverage
    - Experience requirement
    - Education requirement
    - Overall requirement match

Important:
    This is a project-defined explainable matching layer.
    It is separate from the existing ATS score.
"""

import re
from typing import Dict, List, Optional


# ============================================================
# SKILL NORMALIZATION
# ============================================================

SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",

    "ml": "machine learning",
    "dl": "deep learning",

    "tf idf": "tf-idf",

    "node.js": "node js",
    "nodejs": "node js",

    "react.js": "react",
    "reactjs": "react",

    "next.js": "next js",
    "nextjs": "next js",

    "opencv-python": "opencv",
    "cv2": "opencv",

    "postgres": "postgresql",
    "mongo": "mongodb",

    "k8s": "kubernetes"
}


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill to a canonical representation.
    """

    if not skill:
        return ""

    skill = str(
        skill
    ).lower().strip()

    skill = re.sub(
        r"\s+",
        " ",
        skill
    )

    return SKILL_ALIASES.get(
        skill,
        skill
    )


# ============================================================
# SKILL SET
# ============================================================

def create_skill_set(
    skills: List[str]
) -> set:
    """
    Convert a skill list into a normalized set.
    """

    if not skills:
        return set()

    return {
        normalize_skill(skill)
        for skill in skills
        if normalize_skill(skill)
    }


# ============================================================
# SKILL REQUIREMENT COMPARISON
# ============================================================

def compare_required_skills(
    candidate_skills: List[str],
    required_skills: List[str]
) -> Dict:
    """
    Compare candidate skills against required skills.
    """

    candidate_set = create_skill_set(
        candidate_skills
    )

    required_set = create_skill_set(
        required_skills
    )

    if not required_set:

        return {
            "matched": [],
            "missing": [],
            "score": None,
            "total": 0
        }

    matched = sorted(
        candidate_set & required_set
    )

    missing = sorted(
        required_set - candidate_set
    )

    score = (
        len(matched)
        /
        len(required_set)
    ) * 100

    return {
        "matched": matched,
        "missing": missing,
        "score": round(
            score,
            2
        ),
        "total": len(required_set)
    }


# ============================================================
# PREFERRED SKILL COMPARISON
# ============================================================

def compare_preferred_skills(
    candidate_skills: List[str],
    preferred_skills: List[str]
) -> Dict:
    """
    Compare candidate skills against preferred skills.
    """

    candidate_set = create_skill_set(
        candidate_skills
    )

    preferred_set = create_skill_set(
        preferred_skills
    )

    if not preferred_set:

        return {
            "matched": [],
            "missing": [],
            "score": None,
            "total": 0
        }

    matched = sorted(
        candidate_set & preferred_set
    )

    missing = sorted(
        preferred_set - candidate_set
    )

    score = (
        len(matched)
        /
        len(preferred_set)
    ) * 100

    return {
        "matched": matched,
        "missing": missing,
        "score": round(
            score,
            2
        ),
        "total": len(preferred_set)
    }


# ============================================================
# EXPERIENCE MATCHING
# ============================================================

def compare_experience(
    candidate_years: float,
    required_years: float,
    maximum_years: Optional[float] = None
) -> Dict:
    """
    Compare candidate experience against the JD requirement.

    Important:
        Candidate experience comes from explicitly stated
        resume information. It is not inferred from dates.
    """

    try:
        candidate_years = float(
            candidate_years or 0
        )
    except (
        ValueError,
        TypeError
    ):
        candidate_years = 0.0

    try:
        required_years = float(
            required_years or 0
        )
    except (
        ValueError,
        TypeError
    ):
        required_years = 0.0

    # --------------------------------------------------------
    # No JD requirement
    # --------------------------------------------------------

    if required_years <= 0:

        return {
            "candidate_years":
                round(candidate_years, 2),

            "required_years":
                0.0,

            "score":
                None,

            "status":
                "Not specified in Job Description"
        }

    # --------------------------------------------------------
    # Candidate experience unavailable
    # --------------------------------------------------------

    if candidate_years <= 0:

        return {
            "candidate_years":
                0.0,

            "required_years":
                round(required_years, 2),

            "score":
                0.0,

            "status":
                "Experience requirement detected, "
                "but candidate experience was not explicitly detected"
        }

    # --------------------------------------------------------
    # Candidate meets minimum
    # --------------------------------------------------------

    if candidate_years >= required_years:

        status = "Meets stated minimum"

        if (
            maximum_years is not None
            and candidate_years > maximum_years
        ):

            status = (
                "Meets minimum; above stated range"
            )

        return {
            "candidate_years":
                round(candidate_years, 2),

            "required_years":
                round(required_years, 2),

            "score":
                100.0,

            "status":
                status
        }

    # --------------------------------------------------------
    # Candidate below minimum
    # --------------------------------------------------------

    score = (
        candidate_years
        /
        required_years
    ) * 100

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return {
        "candidate_years":
            round(candidate_years, 2),

        "required_years":
            round(required_years, 2),

        "score":
            round(score, 2),

        "status":
            "Below stated minimum"
    }


# ============================================================
# EDUCATION HELPERS
# ============================================================

def detect_degree_level(
    text: str
) -> int:
    """
    Detect approximate highest degree level.

    Levels:

        0 = not detected
        1 = diploma
        2 = bachelor
        3 = master
        4 = doctorate
    """

    if not text:
        return 0

    text = str(
        text
    ).lower()

    # Doctorate
    doctorate_patterns = [
        r"\bph\.?\s*d\.?\b",
        r"\bdoctorate\b",
        r"\bdoctoral\b"
    ]

    if any(
        re.search(
            pattern,
            text
        )
        for pattern in doctorate_patterns
    ):

        return 4

    # Master
    master_patterns = [
        r"\bmaster\b",
        r"\bmaster's\b",
        r"\bm\.?\s*tech\b",
        r"\bm\.?\s*e\.?\b",
        r"\bm\.?\s*sc\b",
        r"\bm\.?\s*ca\b",
        r"\bmba\b",
        r"\bmsc\b",
        r"\bmca\b"
    ]

    if any(
        re.search(
            pattern,
            text
        )
        for pattern in master_patterns
    ):

        return 3

    # Bachelor
    bachelor_patterns = [
        r"\bbachelor\b",
        r"\bbachelor's\b",
        r"\bb\.?\s*tech\b",
        r"\bb\.?\s*e\.?\b",
        r"\bb\.?\s*sc\b",
        r"\bb\.?\s*ca\b",
        r"\bbba\b",
        r"\bbca\b",
        r"\bbsc\b"
    ]

    if any(
        re.search(
            pattern,
            text
        )
        for pattern in bachelor_patterns
    ):

        return 2

    # Diploma
    if re.search(
        r"\bdiploma\b",
        text
    ):

        return 1

    return 0


def detect_required_degree_level(
    education_requirements: List[str]
) -> int:
    """
    Detect the highest degree level mentioned in JD
    education requirements.
    """

    if not education_requirements:
        return 0

    combined_text = " ".join(
        education_requirements
    )

    return detect_degree_level(
        combined_text
    )


def extract_education_keywords(
    text: str
) -> List[str]:
    """
    Extract common education/domain keywords that can help
    compare candidate education with JD requirements.
    """

    if not text:
        return []

    text = str(
        text
    ).lower()

    education_domains = {

        "computer science":
            "computer science",

        "artificial intelligence":
            "artificial intelligence",

        "machine learning":
            "machine learning",

        "data science":
            "data science",

        "information technology":
            "information technology",

        "information systems":
            "information systems",

        "software engineering":
            "software engineering",

        "computer engineering":
            "computer engineering",

        "electronics":
            "electronics",

        "electrical engineering":
            "electrical engineering",

        "engineering":
            "engineering",

        "mathematics":
            "mathematics",

        "statistics":
            "statistics",

        "physics":
            "physics",

        "business":
            "business",

        "management":
            "management"
    }

    found = []

    for keyword, canonical in education_domains.items():

        if keyword in text:

            found.append(
                canonical
            )

    return sorted(
        set(found)
    )


# ============================================================
# EDUCATION MATCHING
# ============================================================

def compare_education(
    candidate_education: str,
    education_requirements: List[str]
) -> Dict:
    """
    Compare candidate education with JD education requirements.

    This is a heuristic comparison based on degree level and
    common study-domain terms.
    """

    if not education_requirements:

        return {
            "score": None,
            "status":
                "Education requirement not specified",
            "candidate_degree_level":
                0,
            "required_degree_level":
                0,
            "domain_match":
                False
        }

    if not candidate_education:

        return {
            "score": 0.0,
            "status":
                "Candidate education not detected",
            "candidate_degree_level":
                0,
            "required_degree_level":
                detect_required_degree_level(
                    education_requirements
                ),
            "domain_match":
                False
        }

    candidate_level = detect_degree_level(
        candidate_education
    )

    required_level = detect_required_degree_level(
        education_requirements
    )

    candidate_domains = set(
        extract_education_keywords(
            candidate_education
        )
    )

    required_domains = set(
        extract_education_keywords(
            " ".join(
                education_requirements
            )
        )
    )

    domain_match = bool(
        candidate_domains
        & required_domains
    )

    # --------------------------------------------------------
    # No degree level could be inferred
    # --------------------------------------------------------

    if required_level == 0:

        return {
            "score": 70.0,
            "status":
                "Education requirement detected, "
                "but degree level could not be classified",
            "candidate_degree_level":
                candidate_level,
            "required_degree_level":
                required_level,
            "domain_match":
                domain_match
        }

    # --------------------------------------------------------
    # Candidate degree below requirement
    # --------------------------------------------------------

    if candidate_level < required_level:

        return {
            "score": 0.0,
            "status":
                "Candidate degree level appears below "
                "the stated requirement",
            "candidate_degree_level":
                candidate_level,
            "required_degree_level":
                required_level,
            "domain_match":
                domain_match
        }

    # --------------------------------------------------------
    # Degree level met
    # --------------------------------------------------------

    if required_domains:

        if domain_match:

            return {
                "score": 100.0,
                "status":
                    "Degree level and study domain detected",
                "candidate_degree_level":
                    candidate_level,
                "required_degree_level":
                    required_level,
                "domain_match":
                    True
            }

        return {
            "score": 70.0,
            "status":
                "Degree level detected; required domain "
                "was not directly detected",
            "candidate_degree_level":
                candidate_level,
            "required_degree_level":
                required_level,
            "domain_match":
                False
        }

    return {
        "score": 100.0,
        "status":
            "Stated degree level detected",
        "candidate_degree_level":
            candidate_level,
        "required_degree_level":
            required_level,
        "domain_match":
            domain_match
    }


# ============================================================
# DEGREE LABELS
# ============================================================

def degree_level_label(
    level: int
) -> str:
    """
    Convert degree level number into readable text.
    """

    labels = {
        0: "Not detected",
        1: "Diploma",
        2: "Bachelor",
        3: "Master",
        4: "Doctorate"
    }

    return labels.get(
        level,
        "Unknown"
    )


# ============================================================
# OVERALL REQUIREMENT SCORE
# ============================================================

def calculate_requirement_match_score(
    required_skill_score,
    preferred_skill_score,
    experience_score,
    education_score
) -> Optional[float]:
    """
    Calculate a project-defined requirement-match score.

    Base weights:

        Required Skills    50%
        Preferred Skills   20%
        Experience         20%
        Education          10%

    Only available components are included and their weights
    are normalized.

    This score is separate from the ATS score.
    """

    components = []

    if required_skill_score is not None:

        components.append(
            (
                float(required_skill_score),
                0.50
            )
        )

    if preferred_skill_score is not None:

        components.append(
            (
                float(preferred_skill_score),
                0.20
            )
        )

    if experience_score is not None:

        components.append(
            (
                float(experience_score),
                0.20
            )
        )

    if education_score is not None:

        components.append(
            (
                float(education_score),
                0.10
            )
        )

    if not components:

        return None

    numerator = sum(
        score * weight
        for score, weight
        in components
    )

    denominator = sum(
        weight
        for _, weight
        in components
    )

    if denominator <= 0:

        return None

    final_score = (
        numerator /
        denominator
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                final_score
            )
        ),
        2
    )


# ============================================================
# MASTER MATCH ANALYSIS
# ============================================================

def analyze_candidate_requirements(
    candidate_intelligence: Dict,
    jd_analysis: Dict
) -> Dict:
    """
    Master function for candidate vs JD requirement analysis.
    """

    if not candidate_intelligence:
        candidate_intelligence = {}

    if not jd_analysis:
        jd_analysis = {}

    candidate_skills = candidate_intelligence.get(
        "all_skills",
        []
    )

    required_skills = jd_analysis.get(
        "required_skills",
        []
    )

    preferred_skills = jd_analysis.get(
        "preferred_skills",
        []
    )

    # --------------------------------------------------------
    # Required skills
    # --------------------------------------------------------

    required = compare_required_skills(
        candidate_skills,
        required_skills
    )

    # --------------------------------------------------------
    # Preferred skills
    # --------------------------------------------------------

    preferred = compare_preferred_skills(
        candidate_skills,
        preferred_skills
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    jd_experience = jd_analysis.get(
        "experience",
        {}
    )

    experience = compare_experience(

        candidate_years=
            candidate_intelligence.get(
                "experience_years",
                0
            ),

        required_years=
            jd_experience.get(
                "minimum_years",
                0
            ),

        maximum_years=
            jd_experience.get(
                "maximum_years"
            )
    )

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    education = compare_education(

        candidate_education=
            candidate_intelligence.get(
                "education",
                ""
            ),

        education_requirements=
            jd_analysis.get(
                "education_requirements",
                []
            )
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    requirement_score = (
        calculate_requirement_match_score(

            required_skill_score=
                required.get(
                    "score"
                ),

            preferred_skill_score=
                preferred.get(
                    "score"
                ),

            experience_score=
                experience.get(
                    "score"
                ),

            education_score=
                education.get(
                    "score"
                )
        )
    )

    return {

        "required_skills":
            required,

        "preferred_skills":
            preferred,

        "experience":
            experience,

        "education":
            education,

        "requirement_match_score":
            requirement_score
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    sample_jd = {

        "required_skills": [
            "python",
            "machine learning",
            "pytorch",
            "sql",
            "tensorflow"
        ],

        "preferred_skills": [
            "docker",
            "aws",
            "fastapi"
        ],

        "experience": {
            "minimum_years": 2.0,
            "maximum_years": None
        },

        "education_requirements": [
            "Bachelor's degree in Computer Science, "
            "Artificial Intelligence or related field."
        ]
    }

    sample_candidate = {

        "all_skills": [
            "python",
            "machine learning",
            "pytorch",
            "sql",
            "docker",
            "fastapi"
        ],

        "experience_years":
            1.5,

        "education":
            "B.Tech in Artificial Intelligence"
    }

    print("=" * 70)
    print("CANDIDATE REQUIREMENT MATCHING ENGINE")
    print("=" * 70)

    result = analyze_candidate_requirements(
        sample_candidate,
        sample_jd
    )

    print("\nREQUIRED SKILLS")
    print("-" * 70)
    print(
        result["required_skills"]
    )

    print("\nPREFERRED SKILLS")
    print("-" * 70)
    print(
        result["preferred_skills"]
    )

    print("\nEXPERIENCE")
    print("-" * 70)
    print(
        result["experience"]
    )

    print("\nEDUCATION")
    print("-" * 70)

    education_result = result[
        "education"
    ]

    print(
        education_result
    )

    print(
        "Candidate Degree:",
        degree_level_label(
            education_result[
                "candidate_degree_level"
            ]
        )
    )

    print(
        "Required Degree:",
        degree_level_label(
            education_result[
                "required_degree_level"
            ]
        )
    )

    print("\nOVERALL REQUIREMENT MATCH")
    print("-" * 70)

    print(
        result[
            "requirement_match_score"
        ]
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)