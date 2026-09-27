
"""
AI Resume Screening Platform
=============================

Resume Quality Engine

Purpose:
    Analyze the overall quality and completeness of a resume.

Checks:
    - Resume length
    - Section completeness
    - Contact information
    - Technical skills
    - Quantified achievements
    - Action verbs
    - Repeated words
    - Keyword stuffing
    - Missing sections
    - Basic extraction/formatting issues

Important:
    This is a resume-quality analysis tool.
    It is NOT a hiring decision or an official ATS score.
"""

import re
from collections import Counter
from typing import Dict, List


# ============================================================
# CONFIGURATION
# ============================================================

IMPORTANT_SECTIONS = [
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications"
]


OPTIONAL_SECTIONS = [
    "achievements",
    "publications",
    "interests"
]


ACTION_VERBS = {
    "achieved",
    "analyzed",
    "automated",
    "built",
    "collaborated",
    "created",
    "decreased",
    "delivered",
    "designed",
    "developed",
    "deployed",
    "engineered",
    "evaluated",
    "implemented",
    "improved",
    "increased",
    "integrated",
    "led",
    "managed",
    "optimized",
    "reduced",
    "researched",
    "solved",
    "tested",
    "trained",
    "developed",
    "maintained",
    "configured",
    "migrated",
    "launched",
    "constructed",
    "processed",
    "predicted",
    "classified",
    "detected",
    "generated",
    "validated"
}


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "been",
    "being",
    "into",
    "their",
    "they",
    "them",
    "than",
    "then",
    "but",
    "not",
    "can",
    "will",
    "would",
    "should",
    "could",
    "may",
    "job",
    "role",
    "work",
    "working",
    "using",
    "used",
    "use",
    "experience",
    "responsible",
    "responsibilities",
    "candidate",
    "team",
    "project",
    "projects",
    "skills",
    "education",
    "university",
    "college"
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize resume text.
    """

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def get_words(text: str) -> List[str]:
    """
    Return normalized alphabetic words.
    """

    if not text:
        return []

    words = re.findall(
        r"\b[a-zA-Z]{2,}\b",
        text.lower()
    )

    return words


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_statistics(
    text: str
) -> Dict[str, int]:
    """
    Calculate basic resume statistics.
    """

    if not text:

        return {
            "characters": 0,
            "words": 0,
            "lines": 0,
            "sentences": 0,
            "bullet_lines": 0
        }

    words = get_words(text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    sentences = re.findall(
        r"[.!?]+",
        text
    )

    bullet_lines = [
        line
        for line in lines
        if re.match(
            r"^[•●▪◦\-*]",
            line
        )
    ]

    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
        "sentences": len(sentences),
        "bullet_lines": len(bullet_lines)
    }


# ============================================================
# SECTION ANALYSIS
# ============================================================

def analyze_sections(
    sections: Dict[str, str]
) -> Dict:
    """
    Analyze important and optional resume sections.
    """

    if not sections:
        sections = {}

    present = []
    missing = []

    for section in IMPORTANT_SECTIONS:

        content = sections.get(
            section,
            ""
        )

        if content and content.strip():

            present.append(section)

        else:

            missing.append(section)

    optional_present = []

    for section in OPTIONAL_SECTIONS:

        content = sections.get(
            section,
            ""
        )

        if content and content.strip():

            optional_present.append(section)

    score = (
        len(present) /
        len(IMPORTANT_SECTIONS)
    ) * 100

    return {
        "present": present,
        "missing": missing,
        "optional_present": optional_present,
        "score": round(score, 2)
    }


# ============================================================
# CONTACT ANALYSIS
# ============================================================

def analyze_contact_information(
    text: str
) -> Dict:
    """
    Check whether important contact information exists.

    This function uses regex only and does not make
    external requests.
    """

    if not text:

        return {
            "email": False,
            "phone": False,
            "linkedin": False,
            "github": False,
            "score": 0.0
        }

    email_pattern = (
        r"\b"
        r"[A-Za-z0-9._%+-]+"
        r"@"
        r"[A-Za-z0-9.-]+"
        r"\."
        r"[A-Za-z]{2,}"
        r"\b"
    )

    phone_patterns = [
        r"\+91[\s.-]?[6-9]\d{4}[\s.-]?\d{5}",
        r"\b[6-9]\d{4}[\s.-]?\d{5}\b",
        r"\+\d{1,3}[\s.-]?\d{7,14}"
    ]

    email_found = bool(
        re.search(
            email_pattern,
            text
        )
    )

    phone_found = False

    for pattern in phone_patterns:

        if re.search(
            pattern,
            text
        ):

            phone_found = True
            break

    linkedin_found = bool(
        re.search(
            r"linkedin\.com/in/",
            text,
            flags=re.IGNORECASE
        )
    )

    github_found = bool(
        re.search(
            r"github\.com/",
            text,
            flags=re.IGNORECASE
        )
    )

    fields = [
        email_found,
        phone_found,
        linkedin_found,
        github_found
    ]

    score = (
        sum(fields) /
        len(fields)
    ) * 100

    return {
        "email": email_found,
        "phone": phone_found,
        "linkedin": linkedin_found,
        "github": github_found,
        "score": round(score, 2)
    }


# ============================================================
# ACTION VERB ANALYSIS
# ============================================================

def find_action_verbs(
    text: str
) -> List[str]:
    """
    Find strong action verbs used in the resume.
    """

    words = get_words(text)

    found = []

    for word in words:

        if word in ACTION_VERBS:

            found.append(word)

    return sorted(
        set(found)
    )


def calculate_action_verb_score(
    text: str
) -> float:
    """
    Estimate usage of action verbs.

    More unique action verbs indicate more evidence
    of achievement-oriented writing.

    This is an informational heuristic.
    """

    verbs = find_action_verbs(
        text
    )

    if not verbs:
        return 0.0

    # Cap the score so a very long resume does not
    # automatically receive a perfect score.
    score = min(
        100,
        len(verbs) * 8
    )

    return round(
        score,
        2
    )


# ============================================================
# QUANTIFIED ACHIEVEMENT ANALYSIS
# ============================================================

def find_quantified_achievements(
    text: str
) -> List[str]:
    """
    Find lines containing measurable information.

    Examples:

        Increased accuracy by 15%
        Reduced processing time by 30%
        Processed 10,000 records
        Improved performance by 2x
        Managed a team of 5
    """

    if not text:
        return []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    patterns = [

        # Percentages
        r"\b\d+(?:\.\d+)?\s*%",

        # Numbers with x
        r"\b\d+(?:\.\d+)?\s*x\b",

        # Currency
        r"[$₹€£]\s?\d+",

        # Large numbers
        r"\b\d{2,}(?:,\d{3})*\b",

        # Years/months
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:years?|months?|weeks?)\b"
    ]

    quantified = []

    for line in lines:

        for pattern in patterns:

            if re.search(
                pattern,
                line,
                flags=re.IGNORECASE
            ):

                quantified.append(
                    line
                )

                break

    return quantified


def calculate_quantification_score(
    text: str
) -> float:
    """
    Calculate a heuristic score for quantified content.
    """

    quantified = find_quantified_achievements(
        text
    )

    if not quantified:
        return 0.0

    score = min(
        100,
        len(quantified) * 15
    )

    return round(
        score,
        2
    )


# ============================================================
# REPETITION ANALYSIS
# ============================================================

def find_repeated_words(
    text: str,
    minimum_count: int = 5
) -> Dict[str, int]:
    """
    Detect frequently repeated non-stop words.
    """

    words = get_words(
        text
    )

    filtered = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    counts = Counter(
        filtered
    )

    repeated = {
        word: count
        for word, count in counts.items()
        if count >= minimum_count
    }

    return dict(
        sorted(
            repeated.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


def calculate_repetition_score(
    text: str
) -> float:
    """
    Calculate a simple repetition score.

    Higher score means fewer suspiciously repeated terms.
    """

    words = get_words(
        text
    )

    if len(words) < 20:

        return 100.0

    repeated = find_repeated_words(
        text
    )

    if not repeated:

        return 100.0

    repeated_occurrences = sum(
        count
        for count in repeated.values()
    )

    ratio = (
        repeated_occurrences /
        len(words)
    )

    score = 100 - (
        ratio * 100
    )

    return round(
        max(0.0, min(100.0, score)),
        2
    )


# ============================================================
# KEYWORD STUFFING ANALYSIS
# ============================================================

def detect_keyword_stuffing(
    text: str
) -> Dict:
    """
    Detect suspicious repetition of technical keywords.

    This is a heuristic check.
    It does not determine whether a resume is fraudulent.
    """

    words = get_words(
        text
    )

    if not words:

        return {
            "detected": False,
            "score": 0.0,
            "repeated_terms": {}
        }

    technical_terms = {
        "python",
        "java",
        "javascript",
        "sql",
        "machine",
        "learning",
        "deep",
        "tensorflow",
        "pytorch",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "react",
        "fastapi",
        "flask",
        "django",
        "nlp",
        "opencv",
        "pandas",
        "numpy",
        "git"
    }

    counts = Counter(
        words
    )

    repeated_terms = {}

    for term in technical_terms:

        count = counts.get(
            term,
            0
        )

        if count >= 8:

            repeated_terms[
                term
            ] = count

    detected = bool(
        repeated_terms
    )

    if not detected:

        score = 0.0

    else:

        score = min(
            100,
            len(repeated_terms) * 20
        )

    return {
        "detected": detected,
        "score": round(score, 2),
        "repeated_terms": dict(
            sorted(
                repeated_terms.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )
    }


# ============================================================
# RESUME LENGTH ANALYSIS
# ============================================================

def analyze_resume_length(
    text: str
) -> Dict:
    """
    Analyze approximate resume length.

    This uses word count rather than PDF page count because
    PDF extraction can vary depending on formatting.
    """

    words = get_words(
        text
    )

    word_count = len(
        words
    )

    if word_count == 0:

        category = "Empty"

    elif word_count < 150:

        category = "Very Short"

    elif word_count < 300:

        category = "Short"

    elif word_count <= 900:

        category = "Typical"

    elif word_count <= 1400:

        category = "Long"

    else:

        category = "Very Long"

    return {
        "word_count": word_count,
        "category": category
    }


# ============================================================
# BULLET / CONTENT STRUCTURE
# ============================================================

def analyze_bullet_structure(
    text: str
) -> Dict:
    """
    Analyze use of bullet points.
    """

    statistics = calculate_statistics(
        text
    )

    lines = statistics[
        "lines"
    ]

    bullets = statistics[
        "bullet_lines"
    ]

    if lines == 0:

        ratio = 0.0

    else:

        ratio = (
            bullets /
            lines
        ) * 100

    return {
        "bullet_lines": bullets,
        "total_lines": lines,
        "bullet_ratio": round(
            ratio,
            2
        )
    }


# ============================================================
# EXTRACTION QUALITY
# ============================================================

def detect_extraction_issues(
    text: str
) -> List[str]:
    """
    Detect obvious PDF/DOCX extraction problems.
    """

    issues = []

    if not text:

        issues.append(
            "No extractable text was detected."
        )

        return issues

    # Excessive replacement characters
    replacement_count = text.count(
        "�"
    )

    if replacement_count >= 3:

        issues.append(
            "Several unsupported or corrupted characters were detected."
        )

    # Excessive broken words
    broken_word_pattern = (
        r"\b[a-zA-Z]{1,2}\s+[a-zA-Z]{1,2}\s+"
        r"[a-zA-Z]{1,2}\b"
    )

    broken_matches = re.findall(
        broken_word_pattern,
        text
    )

    if len(broken_matches) > 10:

        issues.append(
            "The extracted text contains many short fragmented tokens."
        )

    # Very little text
    word_count = len(
        get_words(text)
    )

    if word_count < 50:

        issues.append(
            "Very little readable text was extracted from the document."
        )

    # Repeated strange symbols
    symbol_matches = re.findall(
        r"[|]{3,}|[_]{5,}|[.]{5,}",
        text
    )

    if symbol_matches:

        issues.append(
            "Unusual repeated formatting characters were detected."
        )

    return issues


# ============================================================
# RESUME QUALITY SCORE
# ============================================================

def calculate_quality_score(
    section_score: float,
    contact_score: float,
    action_score: float,
    quantification_score: float,
    repetition_score: float,
    extraction_score: float,
    keyword_stuffing_penalty: float
) -> float:
    """
    Calculate the overall resume quality score.

    Project-defined heuristic:

        Sections          25%
        Contact           15%
        Action verbs      15%
        Quantification    15%
        Repetition        10%
        Extraction        10%
        Keyword hygiene  10%

    Keyword stuffing contributes as a penalty.

    This is NOT an official ATS formula.
    """

    base_score = (

        section_score * 0.25

        + contact_score * 0.15

        + action_score * 0.15

        + quantification_score * 0.15

        + repetition_score * 0.10

        + extraction_score * 0.10

        + (
            (100 - keyword_stuffing_penalty)
            * 0.10
        )
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                base_score
            )
        ),
        2
    )


# ============================================================
# QUALITY STATUS
# ============================================================

def get_quality_status(
    score: float
) -> str:
    """
    Convert quality score into a descriptive category.
    """

    score = float(
        score
    )

    if score >= 80:

        return "Strong Resume Structure"

    if score >= 60:

        return "Good Resume Structure"

    if score >= 40:

        return "Needs Improvement"

    return "Major Improvements Needed"


# ============================================================
# QUALITY RECOMMENDATIONS
# ============================================================

def generate_quality_recommendations(
    section_analysis: Dict,
    contact_analysis: Dict,
    action_score: float,
    quantification_score: float,
    repetition_score: float,
    keyword_analysis: Dict,
    extraction_issues: List[str],
    length_analysis: Dict
) -> List[str]:
    """
    Generate actionable resume-quality recommendations.
    """

    recommendations = []

    # Missing sections
    missing = section_analysis.get(
        "missing",
        []
    )

    if missing:

        readable = ", ".join(
            section.replace(
                "_",
                " "
            ).title()
            for section in missing
        )

        recommendations.append(
            f"Consider adding clear sections for: {readable}."
        )

    # Contact
    if not contact_analysis.get(
        "email",
        False
    ):

        recommendations.append(
            "Add a clearly visible professional email address."
        )

    if not contact_analysis.get(
        "phone",
        False
    ):

        recommendations.append(
            "Add a professional phone number if appropriate."
        )

    if not contact_analysis.get(
        "linkedin",
        False
    ):

        recommendations.append(
            "Consider adding a LinkedIn profile URL."
        )

    if not contact_analysis.get(
        "github",
        False
    ):

        recommendations.append(
            "For technical roles, consider adding a GitHub profile or portfolio."
        )

    # Action verbs
    if action_score < 40:

        recommendations.append(
            "Use stronger action verbs to describe what you built, changed, analyzed or delivered."
        )

    # Quantification
    if quantification_score < 30:

        recommendations.append(
            "Add measurable results where truthful, such as percentages, counts, time saved or performance improvements."
        )

    # Repetition
    if repetition_score < 70:

        recommendations.append(
            "Reduce unnecessary repetition of the same terms and phrases."
        )

    # Keyword stuffing
    if keyword_analysis.get(
        "detected",
        False
    ):

        recommendations.append(
            "Review repeated technical keywords and keep them where they provide genuine evidence."
        )

    # Extraction
    if extraction_issues:

        recommendations.append(
            "Check the original document formatting because some extracted text appears fragmented or corrupted."
        )

    # Length
    length_category = length_analysis.get(
        "category",
        ""
    )

    if length_category == "Very Short":

        recommendations.append(
            "The resume contains relatively little text; ensure important experience, projects and skills are represented."
        )

    elif length_category == "Very Long":

        recommendations.append(
            "Consider removing repetitive or low-value content and prioritizing the most relevant experience."
        )

    if not recommendations:

        recommendations.append(
            "The resume has a solid basic structure. Continue using concise, specific and truthful achievement statements."
        )

    return recommendations


# ============================================================
# MASTER ANALYSIS FUNCTION
# ============================================================

def analyze_resume_quality(
    text: str,
    sections: Dict[str, str] = None
) -> Dict:
    """
    Master Resume Quality function.

    Parameters:
        text:
            Raw extracted resume text.

        sections:
            Optional sections already extracted by
            resume_intelligence.py.

    Returns:
        Complete quality analysis dictionary.
    """

    if not text:

        return {
            "quality_score": 0.0,
            "status": "No Resume Text",

            "statistics": {
                "characters": 0,
                "words": 0,
                "lines": 0,
                "sentences": 0,
                "bullet_lines": 0
            },

            "length": {
                "word_count": 0,
                "category": "Empty"
            },

            "sections": {
                "present": [],
                "missing": IMPORTANT_SECTIONS,
                "optional_present": [],
                "score": 0.0
            },

            "contact": {
                "email": False,
                "phone": False,
                "linkedin": False,
                "github": False,
                "score": 0.0
            },

            "action_verbs": {
                "verbs": [],
                "score": 0.0
            },

            "quantification": {
                "examples": [],
                "score": 0.0
            },

            "repetition": {
                "repeated_words": {},
                "score": 0.0
            },

            "keyword_stuffing": {
                "detected": False,
                "score": 0.0,
                "repeated_terms": {}
            },

            "bullet_structure": {
                "bullet_lines": 0,
                "total_lines": 0,
                "bullet_ratio": 0.0
            },

            "extraction_issues": [
                "No resume text was provided."
            ],

            "recommendations": [
                "Upload a resume containing extractable text."
            ]
        }

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    cleaned_text = clean_text(
        text
    )

    # --------------------------------------------------------
    # Sections
    # --------------------------------------------------------

    if sections is None:

        sections = {}

    section_analysis = analyze_sections(
        sections
    )

    # --------------------------------------------------------
    # Contact
    # --------------------------------------------------------

    contact_analysis = analyze_contact_information(
        cleaned_text
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = calculate_statistics(
        cleaned_text
    )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    length_analysis = analyze_resume_length(
        cleaned_text
    )

    # --------------------------------------------------------
    # Action verbs
    # --------------------------------------------------------

    action_verbs = find_action_verbs(
        cleaned_text
    )

    action_score = calculate_action_verb_score(
        cleaned_text
    )

    # --------------------------------------------------------
    # Quantification
    # --------------------------------------------------------

    quantified_examples = find_quantified_achievements(
        cleaned_text
    )

    quantification_score = calculate_quantification_score(
        cleaned_text
    )

    # --------------------------------------------------------
    # Repetition
    # --------------------------------------------------------

    repeated_words = find_repeated_words(
        cleaned_text
    )

    repetition_score = calculate_repetition_score(
        cleaned_text
    )

    # --------------------------------------------------------
    # Keyword stuffing
    # --------------------------------------------------------

    keyword_analysis = detect_keyword_stuffing(
        cleaned_text
    )

    # --------------------------------------------------------
    # Bullet structure
    # --------------------------------------------------------

    bullet_structure = analyze_bullet_structure(
        cleaned_text
    )

    # --------------------------------------------------------
    # Extraction issues
    # --------------------------------------------------------

    extraction_issues = detect_extraction_issues(
        cleaned_text
    )

    # --------------------------------------------------------
    # Extraction quality score
    # --------------------------------------------------------

    if extraction_issues:

        extraction_score = 70.0

    else:

        extraction_score = 100.0

    # If there is almost no text, score should be low
    if statistics["words"] < 50:

        extraction_score = 20.0

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    quality_score = calculate_quality_score(

        section_score=section_analysis[
            "score"
        ],

        contact_score=contact_analysis[
            "score"
        ],

        action_score=action_score,

        quantification_score=quantification_score,

        repetition_score=repetition_score,

        extraction_score=extraction_score,

        keyword_stuffing_penalty=keyword_analysis[
            "score"
        ]
    )

    status = get_quality_status(
        quality_score
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = generate_quality_recommendations(

        section_analysis=section_analysis,

        contact_analysis=contact_analysis,

        action_score=action_score,

        quantification_score=quantification_score,

        repetition_score=repetition_score,

        keyword_analysis=keyword_analysis,

        extraction_issues=extraction_issues,

        length_analysis=length_analysis
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "quality_score": quality_score,

        "status": status,

        "statistics": statistics,

        "length": length_analysis,

        "sections": section_analysis,

        "contact": contact_analysis,

        "action_verbs": {
            "verbs": action_verbs,
            "score": action_score
        },

        "quantification": {
            "examples": quantified_examples,
            "score": quantification_score
        },

        "repetition": {
            "repeated_words": repeated_words,
            "score": repetition_score
        },

        "keyword_stuffing": keyword_analysis,

        "bullet_structure": bullet_structure,

        "extraction_issues": extraction_issues,

        "recommendations": recommendations
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    sample_resume = """
    Rahul Sharma

    rahul.sharma@gmail.com
    +91 98765 43210
    https://linkedin.com/in/rahulsharma
    https://github.com/rahulsharma

    PROFESSIONAL SUMMARY

    AI undergraduate with experience in Python,
    Machine Learning, NLP and Deep Learning.

    TECHNICAL SKILLS

    Python, C++, SQL, Machine Learning,
    Deep Learning, PyTorch, TensorFlow,
    NLP, OpenCV, Pandas, NumPy,
    Docker, Git, FastAPI.

    EDUCATION

    B.Tech in Artificial Intelligence
    ABC University

    EXPERIENCE

    AI Intern - XYZ Technologies

    • Developed a machine learning pipeline.
    • Improved model accuracy by 18%.
    • Reduced processing time by 30%.
    • Automated data preprocessing for 10,000 records.

    PROJECTS

    • Built an AI Resume Analyzer using NLP.
    • Developed a Mood Places Recommender.
    • Created a Real Time Face Detection system.

    CERTIFICATIONS

    Machine Learning Certification
    Python Certification

    ACHIEVEMENTS

    • Achieved first place in a college hackathon.
    """

    print("=" * 65)
    print("AI RESUME QUALITY ENGINE")
    print("=" * 65)

    # Basic sections for standalone testing
    sample_sections = {

        "summary":
            "AI undergraduate with experience in Python.",

        "skills":
            "Python, SQL, Machine Learning, NLP.",

        "education":
            "B.Tech in Artificial Intelligence.",

        "experience":
            "AI Intern - XYZ Technologies.",

        "projects":
            "AI Resume Analyzer.",

        "certifications":
            "Machine Learning Certification."
    }

    result = analyze_resume_quality(
        sample_resume,
        sample_sections
    )

    print("\nQUALITY SCORE")
    print("-" * 65)

    print(
        result["quality_score"]
    )

    print(
        "Status:",
        result["status"]
    )

    print("\nSTATISTICS")
    print("-" * 65)

    print(
        result["statistics"]
    )

    print("\nRESUME LENGTH")
    print("-" * 65)

    print(
        result["length"]
    )

    print("\nSECTION ANALYSIS")
    print("-" * 65)

    print(
        result["sections"]
    )

    print("\nCONTACT ANALYSIS")
    print("-" * 65)

    print(
        result["contact"]
    )

    print("\nACTION VERBS")
    print("-" * 65)

    print(
        result["action_verbs"]
    )

    print("\nQUANTIFIED ACHIEVEMENTS")
    print("-" * 65)

    for example in result[
        "quantification"
    ]["examples"]:

        print(
            "-",
            example
        )

    print("\nREPETITION")
    print("-" * 65)

    print(
        result["repetition"]
    )

    print("\nKEYWORD STUFFING")
    print("-" * 65)

    print(
        result["keyword_stuffing"]
    )

    print("\nBULLET STRUCTURE")
    print("-" * 65)

    print(
        result["bullet_structure"]
    )

    print("\nEXTRACTION ISSUES")
    print("-" * 65)

    if result["extraction_issues"]:

        for issue in result[
            "extraction_issues"
        ]:

            print(
                "-",
                issue
            )

    else:

        print(
            "No extraction issues detected."
        )

    print("\nRECOMMENDATIONS")
    print("-" * 65)

    for recommendation in result[
        "recommendations"
    ]:

        print(
            "-",
            recommendation
        )

    print("\n" + "=" * 65)
    print("TEST COMPLETE")
    print("=" * 65)
