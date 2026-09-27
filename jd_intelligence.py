"""
AI Resume Screening Platform
============================

Job Description Intelligence Engine

Purpose:
    Convert a raw Job Description into structured information.

Extracts:
    - Required skills
    - Preferred skills
    - Required experience
    - Education requirements
    - Responsibilities
    - General requirements
    - Job sections
    - Skill categories

Design goals:
    - Fast
    - Dependency-light
    - No external API
    - No transformer model
    - Safe for Streamlit
    - Easy integration with ATS engine
"""

import re
from typing import Dict, List


# ============================================================
# SKILL DATABASE
# ============================================================

SKILL_DATABASE = {

    "programming": [
        "python",
        "c++",
        "c#",
        "java",
        "javascript",
        "typescript",
        "sql",
        "golang",
        "go",
        "rust",
        "kotlin",
        "swift",
        "php"
    ],

    "machine_learning": [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "scikit-learn",
        "sklearn",
        "xgboost",
        "lightgbm",
        "random forest",
        "logistic regression",
        "linear regression",
        "decision tree",
        "support vector machine",
        "svm",
        "clustering",
        "k-means",
        "feature engineering",
        "feature selection",
        "model evaluation"
    ],

    "deep_learning": [
        "deep learning",
        "tensorflow",
        "pytorch",
        "keras",
        "cnn",
        "convolutional neural network",
        "rnn",
        "lstm",
        "gru",
        "transformer",
        "transformers",
        "vision transformer",
        "vit",
        "gan",
        "generative adversarial network",
        "autoencoder",
        "diffusion models"
    ],

    "nlp": [
        "nlp",
        "natural language processing",
        "nltk",
        "spacy",
        "bert",
        "gpt",
        "llm",
        "large language model",
        "word2vec",
        "fasttext",
        "tf-idf",
        "text classification",
        "sentiment analysis",
        "named entity recognition",
        "ner",
        "tokenization",
        "text generation",
        "information retrieval"
    ],

    "computer_vision": [
        "opencv",
        "computer vision",
        "image processing",
        "object detection",
        "face detection",
        "yolo",
        "resnet",
        "mtcnn",
        "deepface",
        "image classification",
        "image segmentation",
        "ocr"
    ],

    "data_science": [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "data analysis",
        "data visualization",
        "statistics",
        "data cleaning",
        "exploratory data analysis",
        "eda"
    ],

    "databases": [
        "mysql",
        "postgresql",
        "postgres",
        "mongodb",
        "mongo",
        "sql server",
        "redis",
        "sqlite",
        "oracle",
        "firebase",
        "database"
    ],

    "cloud": [
        "aws",
        "amazon web services",
        "azure",
        "google cloud",
        "gcp",
        "ec2",
        "s3",
        "lambda",
        "cloud computing"
    ],

    "devops": [
        "docker",
        "kubernetes",
        "jenkins",
        "ci/cd",
        "github actions",
        "terraform",
        "ansible",
        "continuous integration",
        "continuous deployment"
    ],

    "web": [
        "html",
        "css",
        "react",
        "react.js",
        "reactjs",
        "node.js",
        "nodejs",
        "fastapi",
        "flask",
        "django",
        "streamlit",
        "next.js",
        "nextjs",
        "rest api",
        "restful api"
    ],

    "tools": [
        "git",
        "github",
        "jira",
        "linux",
        "vscode",
        "visual studio code",
        "postman",
        "jupyter",
        "google colab"
    ]
}


# ============================================================
# SKILL ALIASES
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


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize JD text while preserving useful technical terms.
    """

    if not text:
        return ""

    text = str(text)

    text = text.lower()

    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',

        "node.js": "node js",
        "nodejs": "node js",

        "react.js": "react",
        "reactjs": "react",

        "next.js": "next js",
        "nextjs": "next js",

        "scikit learn": "scikit-learn",
        "sklearn": "scikit-learn",

        "tf idf": "tf-idf"
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
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


# ============================================================
# SAFE TERM MATCHING
# ============================================================

def term_exists(
    term: str,
    text: str
) -> bool:
    """
    Boundary-aware search for a complete skill term.
    """

    if not term or not text:
        return False

    term = normalize_text(
        term
    )

    text = normalize_text(
        text
    )

    if not term:
        return False

    # Ignore one-character "R" because it is too ambiguous
    # for reliable resume/JD extraction.
    if term == "r":
        return False

    pattern = (
        rf"(?<![a-z0-9])"
        rf"{re.escape(term)}"
        rf"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    ) is not None


# ============================================================
# SKILL NORMALIZATION
# ============================================================

def normalize_skill(
    skill: str
) -> str:
    """
    Convert skill aliases into canonical forms.
    """

    if not skill:
        return ""

    skill = normalize_text(
        skill
    )

    return SKILL_ALIASES.get(
        skill,
        skill
    )


# ============================================================
# EXTRACT ALL JD SKILLS
# ============================================================

def extract_jd_skills(
    text: str
) -> Dict[str, List[str]]:
    """
    Detect technical skills throughout the JD.
    """

    if not text:
        return {}

    normalized_text = normalize_text(
        text
    )

    detected = {}

    for category, skills in SKILL_DATABASE.items():

        found = []

        for skill in skills:

            canonical_skill = normalize_skill(
                skill
            )

            if term_exists(
                canonical_skill,
                normalized_text
            ):

                found.append(
                    canonical_skill
                )

        if found:

            detected[
                category
            ] = sorted(
                set(found)
            )

    return detected


# ============================================================
# FLATTEN SKILLS
# ============================================================

def flatten_skills(
    skill_dict: Dict[str, List[str]]
) -> List[str]:
    """
    Flatten categorized skills.
    """

    skills = []

    if not skill_dict:
        return skills

    for values in skill_dict.values():

        skills.extend(
            values
        )

    return sorted(
        set(skills)
    )


# ============================================================
# SECTION DETECTION
# ============================================================

SECTION_ALIASES = {

    "summary": [
        "summary",
        "job summary",
        "role summary",
        "position summary",
        "about the role",
        "about this role"
    ],

    "responsibilities": [
        "responsibilities",
        "key responsibilities",
        "job responsibilities",
        "responsibilities and duties",
        "duties",
        "what you'll do",
        "what you will do",
        "role and responsibilities"
    ],

    "requirements": [
        "requirements",
        "job requirements",
        "required qualifications",
        "required skills",
        "basic qualifications",
        "minimum qualifications",
        "qualifications"
    ],

    "preferred": [
        "preferred qualifications",
        "preferred skills",
        "preferred requirements",
        "desired qualifications",
        "desired skills",
        "nice to have",
        "nice-to-have",
        "preferred"
    ],

    "education": [
        "education",
        "educational requirements",
        "education requirements",
        "academic requirements"
    ],

    "experience": [
        "experience",
        "required experience",
        "professional experience",
        "work experience"
    ]
}


def normalize_heading(
    text: str
) -> str:
    """
    Normalize a possible JD heading.
    """

    if not text:
        return ""

    text = text.lower().strip()

    text = text.replace(
        "&",
        "and"
    )

    text = re.sub(
        r"[:|•\-]+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def detect_jd_section(
    line: str
) -> str:
    """
    Detect JD section heading.
    """

    normalized = normalize_heading(
        line
    )

    if not normalized:
        return ""

    for section, aliases in SECTION_ALIASES.items():

        for alias in aliases:

            if normalized == normalize_heading(
                alias
            ):

                return section

    return ""


# ============================================================
# EXTRACT JD SECTIONS
# ============================================================

def extract_jd_sections(
    text: str
) -> Dict[str, str]:
    """
    Extract structured sections from a Job Description.
    """

    if not text:
        return {}

    lines = text.splitlines()

    sections = {}

    current_section = None
    buffer = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        detected = detect_jd_section(
            line
        )

        if detected:

            if current_section and buffer:

                content = "\n".join(
                    buffer
                ).strip()

                if content:

                    if current_section in sections:

                        sections[
                            current_section
                        ] += (
                            "\n" + content
                        )

                    else:

                        sections[
                            current_section
                        ] = content

            current_section = detected
            buffer = []

            continue

        if current_section:

            buffer.append(
                line
            )

    # Save final section
    if current_section and buffer:

        content = "\n".join(
            buffer
        ).strip()

        if content:

            if current_section in sections:

                sections[
                    current_section
                ] += (
                    "\n" + content
                )

            else:

                sections[
                    current_section
                ] = content

    return sections


# ============================================================
# REQUIRED / PREFERRED TEXT DETECTION
# ============================================================

REQUIRED_MARKERS = [
    "required",
    "must",
    "mandatory",
    "minimum",
    "essential",
    "need to have",
    "should have",
    "you have"
]


PREFERRED_MARKERS = [
    "preferred",
    "nice to have",
    "nice-to-have",
    "desired",
    "bonus",
    "plus",
    "advantage",
    "good to have"
]


def contains_marker(
    text: str,
    markers: List[str]
) -> bool:
    """
    Check whether text contains one of the supplied
    requirement markers.
    """

    if not text:
        return False

    lower_text = text.lower()

    return any(
        marker in lower_text
        for marker in markers
    )


# ============================================================
# SENTENCE-LEVEL SKILL CLASSIFICATION
# ============================================================

def classify_skill_requirements(
    text: str
) -> Dict[str, List[str]]:
    """
    Classify detected skills as:

        required
        preferred
        unspecified

    This is based on nearby language and section context.

    It is heuristic and should not be treated as a
    definitive legal/employment requirement classifier.
    """

    result = {
        "required": [],
        "preferred": [],
        "unspecified": []
    }

    if not text:
        return result

    normalized_text = normalize_text(
        text
    )

    all_skills = flatten_skills(
        extract_jd_skills(
            normalized_text
        )
    )

    # Break into rough sentences / lines.
    fragments = re.split(
        r"[\n.;•●▪]+",
        normalized_text
    )

    for skill in all_skills:

        matched_fragments = []

        for fragment in fragments:

            if term_exists(
                skill,
                fragment
            ):

                matched_fragments.append(
                    fragment
                )

        if not matched_fragments:

            result[
                "unspecified"
            ].append(skill)

            continue

        skill_class = "unspecified"

        for fragment in matched_fragments:

            if contains_marker(
                fragment,
                PREFERRED_MARKERS
            ):

                skill_class = "preferred"
                break

            if contains_marker(
                fragment,
                REQUIRED_MARKERS
            ):

                skill_class = "required"

        result[
            skill_class
        ].append(skill)

    # Remove duplicates
    for category in result:

        result[
            category
        ] = sorted(
            set(
                result[category]
            )
        )

    return result


# ============================================================
# REQUIRED AND PREFERRED SKILL EXTRACTION
# ============================================================

def extract_required_preferred_skills(
    text: str,
    sections: Dict[str, str] = None
) -> Dict:
    """
    Extract required and preferred skills.

    Strategy:
        1. Analyze explicit Preferred section.
        2. Analyze Requirements section.
        3. Analyze remaining JD text.
        4. Avoid placing the same skill in both lists.

    """

    if not text:
        return {
            "required": [],
            "preferred": [],
            "unspecified": []
        }

    if sections is None:
        sections = extract_jd_sections(
            text
        )

    all_skills = flatten_skills(
        extract_jd_skills(
            text
        )
    )

    required = set()
    preferred = set()
    unspecified = set()

    # --------------------------------------------------------
    # Explicit preferred section
    # --------------------------------------------------------

    preferred_text = sections.get(
        "preferred",
        ""
    )

    if preferred_text:

        preferred_skills = flatten_skills(
            extract_jd_skills(
                preferred_text
            )
        )

        preferred.update(
            preferred_skills
        )

    # --------------------------------------------------------
    # Explicit requirements section
    # --------------------------------------------------------

    requirements_text = sections.get(
        "requirements",
        ""
    )

    if requirements_text:

        classified = classify_skill_requirements(
            requirements_text
        )

        required.update(
            classified["required"]
        )

        preferred.update(
            classified["preferred"]
        )

        unspecified.update(
            classified["unspecified"]
        )

    # --------------------------------------------------------
    # Analyze full JD
    # --------------------------------------------------------

    classified_full = classify_skill_requirements(
        text
    )

    required.update(
        classified_full["required"]
    )

    preferred.update(
        classified_full["preferred"]
    )

    unspecified.update(
        classified_full["unspecified"]
    )

    # --------------------------------------------------------
    # If no explicit marker exists, skill is unspecified.
    # --------------------------------------------------------

    known_skills = (
        required
        | preferred
    )

    unspecified = {
        skill
        for skill in (
            unspecified
            | set(all_skills)
        )
        if skill not in known_skills
    }

    # --------------------------------------------------------
    # Explicit preferred wins over required.
    # This prevents contradictory duplicates.
    # --------------------------------------------------------

    required -= preferred

    unspecified -= (
        required
        | preferred
    )

    return {
        "required":
            sorted(required),

        "preferred":
            sorted(preferred),

        "unspecified":
            sorted(unspecified)
    }


# ============================================================
# EXPERIENCE REQUIREMENT
# ============================================================

def extract_required_experience(
    text: str
) -> Dict:
    """
    Extract the highest explicitly stated experience
    requirement from the JD.

    Examples:

        2 years experience
        3+ years of experience
        minimum 5 years
        1-2 years experience

    Returns the maximum stated lower-bound style value
    when multiple requirements exist.
    """

    if not text:

        return {
            "minimum_years": 0.0,
            "maximum_years": None,
            "raw_matches": []
        }

    patterns = [

        # 3+ years experience
        r"(\d+(?:\.\d+)?)\s*\+?\s*"
        r"(?:years?|yrs?)"
        r"(?:\s+of)?\s+"
        r"(?:professional\s+|relevant\s+|work\s+)?"
        r"experience",

        # experience of 3 years
        r"experience"
        r"\s+(?:of\s+)?"
        r"(\d+(?:\.\d+)?)\s*\+?\s*"
        r"(?:years?|yrs?)",

        # minimum 3 years
        r"(?:minimum|min\.?|at least)"
        r"\s+"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:years?|yrs?)"
    ]

    minimum_values = []
    raw_matches = []

    for pattern in patterns:

        matches = re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            raw = match.group(
                0
            ).strip()

            raw_matches.append(
                raw
            )

            try:

                value = float(
                    match.group(1)
                )

                if 0 <= value <= 50:

                    minimum_values.append(
                        value
                    )

            except (
                ValueError,
                TypeError
            ):

                continue

    minimum_years = (
        max(minimum_values)
        if minimum_values
        else 0.0
    )

    # Detect explicit experience ranges such as 1-3 years.
    range_pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"-\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:years?|yrs?)"
    )

    range_matches = re.findall(
        range_pattern,
        text,
        flags=re.IGNORECASE
    )

    maximum_years = None

    if range_matches:

        range_max_values = []

        for _, max_value in range_matches:

            try:

                value = float(
                    max_value
                )

                if 0 <= value <= 50:

                    range_max_values.append(
                        value
                    )

            except (
                ValueError,
                TypeError
            ):

                continue

        if range_max_values:

            maximum_years = max(
                range_max_values
            )

            # Use range lower bound if available.
            range_min_values = []

            for min_value, _ in range_matches:

                try:

                    value = float(
                        min_value
                    )

                    if 0 <= value <= 50:

                        range_min_values.append(
                            value
                        )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

            if range_min_values:

                minimum_years = max(
                    minimum_years,
                    max(range_min_values)
                )

    return {
        "minimum_years":
            round(
                minimum_years,
                2
            ),

        "maximum_years":
            (
                round(
                    maximum_years,
                    2
                )
                if maximum_years is not None
                else None
            ),

        "raw_matches":
            sorted(
                set(raw_matches)
            )
    }


# ============================================================
# EDUCATION REQUIREMENTS
# ============================================================

EDUCATION_TERMS = [
    "bachelor",
    "bachelor's",
    "b tech",
    "b.tech",
    "b.e",
    "bca",
    "bsc",
    "master",
    "master's",
    "m tech",
    "m.tech",
    "m.e",
    "mca",
    "msc",
    "mba",
    "phd",
    "ph.d",
    "degree",
    "diploma"
]


def extract_education_requirements(
    text: str
) -> List[str]:
    """
    Extract education-related lines.
    """

    if not text:
        return []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    education_lines = []

    for line in lines:

        lower_line = line.lower()

        if any(
            term in lower_line
            for term in EDUCATION_TERMS
        ):

            education_lines.append(
                line
            )

    return sorted(
        set(education_lines)
    )


# ============================================================
# RESPONSIBILITY EXTRACTION
# ============================================================

RESPONSIBILITY_MARKERS = [
    "responsible for",
    "you will",
    "you'll",
    "develop",
    "design",
    "build",
    "implement",
    "maintain",
    "create",
    "analyze",
    "deploy",
    "collaborate",
    "manage",
    "lead",
    "test",
    "optimize",
    "integrate",
    "monitor"
]


def extract_responsibilities(
    text: str,
    sections: Dict[str, str] = None
) -> List[str]:
    """
    Extract likely responsibility lines.
    """

    if not text:
        return []

    if sections is None:

        sections = extract_jd_sections(
            text
        )

    responsibility_section = sections.get(
        "responsibilities",
        ""
    )

    # Prefer explicit responsibilities section
    if responsibility_section:

        lines = [
            line.strip()
            for line
            in responsibility_section.splitlines()
            if line.strip()
        ]

        return lines

    # Fallback: infer responsibility-like lines
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    responsibilities = []

    for line in lines:

        lower_line = line.lower()

        if any(
            marker in lower_line
            for marker in RESPONSIBILITY_MARKERS
        ):

            responsibilities.append(
                line
            )

    return sorted(
        set(responsibilities)
    )


# ============================================================
# GENERAL REQUIREMENTS
# ============================================================

def extract_general_requirements(
    text: str,
    sections: Dict[str, str] = None
) -> List[str]:
    """
    Extract requirements that are not specifically
    technical skills.
    """

    if not text:
        return []

    if sections is None:

        sections = extract_jd_sections(
            text
        )

    requirements = sections.get(
        "requirements",
        ""
    )

    if not requirements:

        return []

    lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip()
    ]

    return lines


# ============================================================
# SKILL CATEGORY SUMMARY
# ============================================================

def summarize_skill_categories(
    skills: Dict[str, List[str]]
) -> Dict[str, int]:
    """
    Count skills by category.
    """

    result = {}

    if not skills:
        return result

    for category, values in skills.items():

        result[
            category
        ] = len(values)

    return result


# ============================================================
# MASTER JD ANALYSIS
# ============================================================

def analyze_job_description(
    text: str
) -> Dict:
    """
    Master Job Description Intelligence function.

    Returns structured JD intelligence that can be used by
    the ATS engine and recruiter dashboard.
    """

    if not text:

        return {

            "skills": {},

            "all_skills": [],

            "skill_categories": {},

            "required_skills": [],

            "preferred_skills": [],

            "unspecified_skills": [],

            "experience": {
                "minimum_years": 0.0,
                "maximum_years": None,
                "raw_matches": []
            },

            "education_requirements": [],

            "responsibilities": [],

            "general_requirements": [],

            "sections": {}
        }

    cleaned_text = normalize_text(
        text
    )

    sections = extract_jd_sections(
        cleaned_text
    )

    skills = extract_jd_skills(
        cleaned_text
    )

    all_skills = flatten_skills(
        skills
    )

    skill_classification = (
        extract_required_preferred_skills(
            cleaned_text,
            sections
        )
    )

    experience = extract_required_experience(
        cleaned_text
    )

    education_requirements = (
        extract_education_requirements(
            cleaned_text
        )
    )

    responsibilities = (
        extract_responsibilities(
            cleaned_text,
            sections
        )
    )

    general_requirements = (
        extract_general_requirements(
            cleaned_text,
            sections
        )
    )

    skill_categories = (
        summarize_skill_categories(
            skills
        )
    )

    return {

        "skills":
            skills,

        "all_skills":
            all_skills,

        "skill_categories":
            skill_categories,

        "required_skills":
            skill_classification[
                "required"
            ],

        "preferred_skills":
            skill_classification[
                "preferred"
            ],

        "unspecified_skills":
            skill_classification[
                "unspecified"
            ],

        "experience":
            experience,

        "education_requirements":
            education_requirements,

        "responsibilities":
            responsibilities,

        "general_requirements":
            general_requirements,

        "sections":
            sections
    }


# ============================================================
# CANDIDATE JD MATCH SUMMARY
# ============================================================

def compare_candidate_to_jd(
    candidate_skills: List[str],
    jd_analysis: Dict
) -> Dict:
    """
    Compare candidate skills with the JD's required,
    preferred and unspecified skills.

    This does NOT calculate the final ATS score.
    """

    candidate_set = {
        normalize_skill(skill)
        for skill in candidate_skills
    }

    required = {
        normalize_skill(skill)
        for skill in jd_analysis.get(
            "required_skills",
            []
        )
    }

    preferred = {
        normalize_skill(skill)
        for skill in jd_analysis.get(
            "preferred_skills",
            []
        )
    }

    unspecified = {
        normalize_skill(skill)
        for skill in jd_analysis.get(
            "unspecified_skills",
            []
        )
    }

    matched_required = sorted(
        candidate_set & required
    )

    missing_required = sorted(
        required - candidate_set
    )

    matched_preferred = sorted(
        candidate_set & preferred
    )

    missing_preferred = sorted(
        preferred - candidate_set
    )

    matched_unspecified = sorted(
        candidate_set & unspecified
    )

    return {

        "matched_required":
            matched_required,

        "missing_required":
            missing_required,

        "matched_preferred":
            matched_preferred,

        "missing_preferred":
            missing_preferred,

        "matched_unspecified":
            matched_unspecified
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    sample_jd = """
    AI Engineer

    ABOUT THE ROLE

    We are looking for an AI Engineer to build and deploy
    machine learning applications.

    REQUIREMENTS

    Python is required.
    Strong machine learning and deep learning knowledge is required.
    Experience with PyTorch and TensorFlow is required.
    SQL experience is required.
    Minimum 2 years of professional experience.
    Bachelor's degree in Computer Science, Artificial
    Intelligence or a related field.

    PREFERRED QUALIFICATIONS

    Experience with Docker and AWS is preferred.
    Knowledge of NLP and FastAPI is a plus.
    GitHub Actions is nice to have.

    RESPONSIBILITIES

    Develop machine learning systems.
    Build and deploy AI applications.
    Design APIs using FastAPI.
    Analyze model performance.
    Collaborate with engineering teams.
    """

    print("=" * 70)
    print("JOB DESCRIPTION INTELLIGENCE ENGINE")
    print("=" * 70)

    result = analyze_job_description(
        sample_jd
    )

    print("\nALL SKILLS")
    print("-" * 70)

    print(
        result["all_skills"]
    )

    print("\nSKILLS BY CATEGORY")
    print("-" * 70)

    for category, skills in result["skills"].items():

        print(
            f"{category}: {skills}"
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

    print("\nUNSPECIFIED SKILLS")
    print("-" * 70)

    print(
        result["unspecified_skills"]
    )

    print("\nEXPERIENCE")
    print("-" * 70)

    print(
        result["experience"]
    )

    print("\nEDUCATION")
    print("-" * 70)

    for item in result[
        "education_requirements"
    ]:

        print(
            "-",
            item
        )

    print("\nRESPONSIBILITIES")
    print("-" * 70)

    for item in result[
        "responsibilities"
    ]:

        print(
            "-",
            item
        )

    print("\nSECTIONS")
    print("-" * 70)

    print(
        list(
            result[
                "sections"
            ].keys()
        )
    )

    print("\nCANDIDATE COMPARISON")
    print("-" * 70)

    candidate_skills = [
        "python",
        "machine learning",
        "pytorch",
        "sql",
        "docker",
        "fastapi"
    ]

    comparison = compare_candidate_to_jd(
        candidate_skills,
        result
    )

    print(
        comparison
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)