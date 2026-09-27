"""
AI Resume Screening Platform
============================

Core Analysis Engine

Features:
    - Text normalization
    - TF-IDF lexical similarity
    - Batch TF-IDF similarity
    - JD keyword extraction
    - Boundary-aware keyword matching
    - Technical skill extraction
    - Skill aliases
    - Category-wise skill analysis
    - Skill gap analysis
    - Section-aware semantic matching
    - Batch semantic matching
    - Complete resume analysis

Performance:
    - Lexical similarity is calculated for all resumes in one
      TF-IDF operation.
    - Semantic embeddings are generated in a single batch.
    - Duplicate semantic texts are embedded only once.
"""

from typing import List, Tuple, Dict

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from semantic_matcher import (
    compute_semantic_similarity,
    compute_section_aware_similarity,
    generate_embeddings,
    prepare_text,
    cosine_score,
    SECTION_WEIGHTS
)

from resume_utils import (
    extract_resume_sections,
    extract_jd_sections
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text while preserving useful technical symbols.
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

        "tf idf": "tf-idf",

        "machine-learning":
            "machine learning",

        "deep-learning":
            "deep learning"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    text = re.sub(
        r"[^a-z0-9+#./\-\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# BOUNDARY-AWARE TERM MATCHING
# ============================================================

def term_exists(
    term: str,
    text: str
) -> bool:
    """
    Check whether a complete technical term exists.

    Helps avoid:
        java  -> javascript
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
# SINGLE RESUME TF-IDF SIMILARITY
# ============================================================

def compute_similarity(
    jd_clean: str,
    resume_clean: str
) -> float:
    """
    Calculate TF-IDF cosine similarity for one resume.
    """

    if not jd_clean or not resume_clean:
        return 0.0

    jd_clean = normalize_text(
        jd_clean
    )

    resume_clean = normalize_text(
        resume_clean
    )

    if not jd_clean or not resume_clean:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:

        vectors = vectorizer.fit_transform(
            [
                jd_clean,
                resume_clean
            ]
        )

    except ValueError:

        return 0.0

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]

    return round(
        float(similarity * 100),
        2
    )


# ============================================================
# BATCH TF-IDF SIMILARITY
# ============================================================

def compute_lexical_scores_batch(
    jd_text: str,
    resume_texts: List[str]
) -> List[float]:
    """
    Calculate TF-IDF similarity for all resumes in one
    vectorization operation.

    This is considerably more efficient than fitting a new
    TfidfVectorizer for every resume.
    """

    if not jd_text or not resume_texts:

        return [
            0.0
            for _ in resume_texts
        ]

    documents = [
        normalize_text(jd_text)
    ]

    documents.extend(
        normalize_text(text)
        for text in resume_texts
    )

    if not documents[0]:

        return [
            0.0
            for _ in resume_texts
        ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:

        matrix = vectorizer.fit_transform(
            documents
        )

    except ValueError:

        return [
            0.0
            for _ in resume_texts
        ]

    jd_vector = matrix[0]

    resume_matrix = matrix[
        1:
    ]

    if resume_matrix.shape[0] == 0:

        return []

    scores = cosine_similarity(
        jd_vector,
        resume_matrix
    )[0]

    return [
        round(
            max(
                0.0,
                min(
                    100.0,
                    float(score) * 100
                )
            ),
            2
        )
        for score in scores
    ]


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_top_keywords(
    jd_clean: str,
    n: int = 20
) -> List[str]:
    """
    Extract important JD keywords using frequency,
    bigram weighting and technical skill boosts.

    Note:
        A single JD does not provide a meaningful multi-document
        IDF distribution, so this is intentionally a weighted
        lexical keyword extractor rather than claiming to be
        classic corpus-level TF-IDF ranking.
    """

    if not jd_clean:
        return []

    text = normalize_text(
        jd_clean
    )

    tokens = re.findall(
        r"[a-zA-Z0-9+#.-]+",
        text
    )

    if not tokens:
        return []

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "your",
        "our",
        "you",
        "are",
        "will",
        "have",
        "has",
        "been",
        "being",
        "into",
        "their",
        "they",
        "them",
        "than",
        "then",
        "must",
        "should",
        "would",
        "could",
        "can",
        "may",
        "job",
        "role",
        "work",
        "team",
        "using",
        "use",
        "required",
        "requirements",
        "candidate",
        "experience"
    }

    filtered_tokens = [
        token
        for token in tokens
        if token.lower() not in stop_words
        and len(token) > 1
    ]

    unigram_counts = Counter(
        filtered_tokens
    )

    bigrams = []

    for i in range(
        len(filtered_tokens) - 1
    ):

        bigrams.append(
            f"{filtered_tokens[i]} "
            f"{filtered_tokens[i + 1]}"
        )

    bigram_counts = Counter(
        bigrams
    )

    candidates = {}

    # Unigram score
    for word, count in unigram_counts.items():

        candidates[word] = (
            count * 1.0
        )

    # Bigram score
    for phrase, count in bigram_counts.items():

        candidates[phrase] = (
            count * 1.5
        )

    # Technical skill boost
    all_skills = flatten_skills(
        extract_skills(
            jd_clean
        )
    )

    for skill in all_skills:

        normalized_skill = normalize_skill(
            skill
        )

        if term_exists(
            normalized_skill,
            text
        ):

            candidates[
                normalized_skill
            ] = (
                candidates.get(
                    normalized_skill,
                    0
                )
                + 5.0
            )

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            item[1],
            len(item[0])
        ),
        reverse=True
    )

    results = []

    for keyword, _ in ranked:

        keyword = keyword.strip()

        if not keyword:
            continue

        duplicate = False

        for existing in results:

            if keyword == existing:

                duplicate = True
                break

            if (
                len(keyword) > 4
                and keyword in existing
            ):

                duplicate = True
                break

        if duplicate:
            continue

        results.append(
            keyword
        )

        if len(results) >= n:
            break

    return results


# ============================================================
# KEYWORD MATCHING
# ============================================================

def find_keyword_matches(
    jd_clean: str,
    resume_clean: str,
    keywords: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Find matched and missing JD keywords.
    """

    jd_clean = normalize_text(
        jd_clean
    )

    resume_clean = normalize_text(
        resume_clean
    )

    matched = []
    missing = []

    for keyword in keywords:

        keyword = normalize_text(
            keyword
        )

        if not keyword:
            continue

        if not term_exists(
            keyword,
            jd_clean
        ):
            continue

        if term_exists(
            keyword,
            resume_clean
        ):

            matched.append(
                keyword
            )

        else:

            missing.append(
                keyword
            )

    return (
        matched,
        missing
    )


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
        "r",
        "go",
        "rust",
        "kotlin",
        "swift"
    ],

    "machine_learning": [
        "machine learning",
        "deep learning",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "scikit-learn",
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
        "feature engineering"
    ],

    "deep_learning": [
        "tensorflow",
        "pytorch",
        "keras",
        "cnn",
        "rnn",
        "lstm",
        "gru",
        "transformer",
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
        "transformers"
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
        "mongodb",
        "sql server",
        "database",
        "redis",
        "sqlite",
        "oracle",
        "firebase"
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
        "node js",
        "fastapi",
        "flask",
        "django",
        "streamlit",
        "next js",
        "rest api",
        "restful api"
    ],

    "tools": [
        "git",
        "github",
        "jira",
        "linux",
        "vscode",
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

    "tf idf": "tf-idf",

    "node.js": "node js",

    "nodejs": "node js",

    "react.js": "react",

    "reactjs": "react",

    "next.js": "next js",

    "nextjs": "next js",

    "opencv-python": "opencv",

    "cv2": "opencv",

    "ml": "machine learning",

    "dl": "deep learning",

    "k8s": "kubernetes",

    "postgres": "postgresql",

    "mongo": "mongodb"
}


def normalize_skill(
    skill: str
) -> str:
    """
    Normalize skill aliases.
    """

    skill = normalize_text(
        skill
    )

    return SKILL_ALIASES.get(
        skill,
        skill
    )


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(
    text: str
) -> Dict[str, List[str]]:
    """
    Extract technical skills grouped by category.
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

            normalized_skill = normalize_skill(
                skill
            )

            if term_exists(
                normalized_skill,
                normalized_text
            ):

                found.append(
                    normalized_skill
                )

        if found:

            detected[
                category
            ] = sorted(
                set(found)
            )

    return detected


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
# SKILL COMPARISON
# ============================================================

def compare_skills(
    jd_text: str,
    resume_text: str
) -> Tuple[List[str], List[str]]:
    """
    Compare JD skills against resume skills.
    """

    jd_skills = flatten_skills(
        extract_skills(
            jd_text
        )
    )

    resume_skills = flatten_skills(
        extract_skills(
            resume_text
        )
    )

    resume_set = {
        normalize_skill(skill)
        for skill in resume_skills
    }

    matched = []
    missing = []

    for skill in jd_skills:

        normalized_skill = normalize_skill(
            skill
        )

        if normalized_skill in resume_set:

            matched.append(
                skill
            )

        else:

            missing.append(
                skill
            )

    return (
        sorted(set(matched)),
        sorted(set(missing))
    )


def compare_skills_by_category(
    jd_text: str,
    resume_text: str
) -> Dict:
    """
    Compare skills category by category.
    """

    jd_skills = extract_skills(
        jd_text
    )

    resume_skills = extract_skills(
        resume_text
    )

    categories = {}

    for category, jd_values in jd_skills.items():

        resume_values = resume_skills.get(
            category,
            []
        )

        normalized_resume = {
            normalize_skill(skill)
            for skill in resume_values
        }

        matched = []
        missing = []

        for skill in jd_values:

            normalized_skill = normalize_skill(
                skill
            )

            if normalized_skill in normalized_resume:

                matched.append(
                    skill
                )

            else:

                missing.append(
                    skill
                )

        total = (
            len(matched)
            + len(missing)
        )

        if total:

            score = (
                len(matched)
                / total
            ) * 100

        else:

            score = 0.0

        categories[
            category
        ] = {

            "matched":
                sorted(set(matched)),

            "missing":
                sorted(set(missing)),

            "score":
                round(
                    score,
                    2
                )
        }

    return categories


# ============================================================
# SKILL SCORE
# ============================================================

def calculate_skill_match_score(
    jd_text: str,
    resume_text: str
) -> float:
    """
    Calculate overall skill match percentage.
    """

    matched, missing = compare_skills(
        jd_text,
        resume_text
    )

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


def get_skill_statistics(
    jd_text: str,
    resume_text: str
) -> Dict:
    """
    Return skill statistics.
    """

    jd_skills = extract_skills(
        jd_text
    )

    resume_skills = extract_skills(
        resume_text
    )

    matched, missing = compare_skills(
        jd_text,
        resume_text
    )

    return {

        "jd_skill_count":
            len(
                flatten_skills(
                    jd_skills
                )
            ),

        "resume_skill_count":
            len(
                flatten_skills(
                    resume_skills
                )
            ),

        "matched_skill_count":
            len(matched),

        "missing_skill_count":
            len(missing),

        "skill_match_score":
            calculate_skill_match_score(
                jd_text,
                resume_text
            ),

        "jd_categories":
            list(
                jd_skills.keys()
            ),

        "resume_categories":
            list(
                resume_skills.keys()
            )
    }


# ============================================================
# SECTION HELPERS
# ============================================================

def analyze_section_semantic_match(
    jd_text: str,
    resume_text: str
) -> Dict:
    """
    Compatibility wrapper around the section-aware
    semantic engine.
    """

    try:

        jd_sections = extract_jd_sections(
            jd_text
        )

    except Exception:

        jd_sections = {}

    try:

        resume_sections = extract_resume_sections(
            resume_text
        )

    except Exception:

        resume_sections = {}

    result = compute_section_aware_similarity(

        jd_sections=jd_sections,

        resume_sections=resume_sections,

        fallback_jd=jd_text,

        fallback_resume=resume_text
    )

    return {

        "section_aware_score":
            result.get(
                "section_aware_score",
                0.0
            ),

        "section_scores":
            result.get(
                "section_scores",
                {}
            ),

        "sections_compared":
            result.get(
                "sections_compared",
                []
            ),

        "jd_sections":
            list(
                jd_sections.keys()
            ),

        "resume_sections":
            list(
                resume_sections.keys()
            )
    }


# ============================================================
# BUILD SECTION COMPARISONS
# ============================================================

def _build_section_comparisons(
    jd_sections: Dict[str, str],
    resume_sections: Dict[str, str]
) -> List[Dict]:
    """
    Build section comparison definitions.

    The structure mirrors semantic_matcher.py so the batch
    engine produces the same conceptual section-aware score.
    """

    comparisons = []

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    jd_skills = jd_sections.get(
        "skills",
        ""
    )

    resume_skills = resume_sections.get(
        "skills",
        ""
    )

    if jd_skills and resume_skills:

        comparisons.append({

            "section":
                "skills",

            "jd":
                jd_skills,

            "resume":
                resume_skills,

            "weight":
                SECTION_WEIGHTS[
                    "skills"
                ]
        })

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    jd_experience_parts = [

        jd_sections.get(
            "experience",
            ""
        ),

        jd_sections.get(
            "responsibilities",
            ""
        ),

        jd_sections.get(
            "requirements",
            ""
        )
    ]

    jd_experience = " ".join(
        part
        for part in jd_experience_parts
        if part
    ).strip()

    resume_experience = resume_sections.get(
        "experience",
        ""
    )

    if jd_experience and resume_experience:

        comparisons.append({

            "section":
                "experience",

            "jd":
                jd_experience,

            "resume":
                resume_experience,

            "weight":
                SECTION_WEIGHTS[
                    "experience"
                ]
        })

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    jd_project_parts = [

        jd_sections.get(
            "responsibilities",
            ""
        ),

        jd_sections.get(
            "requirements",
            ""
        )
    ]

    jd_projects = " ".join(
        part
        for part in jd_project_parts
        if part
    ).strip()

    resume_projects = resume_sections.get(
        "projects",
        ""
    )

    if jd_projects and resume_projects:

        comparisons.append({

            "section":
                "projects",

            "jd":
                jd_projects,

            "resume":
                resume_projects,

            "weight":
                SECTION_WEIGHTS[
                    "projects"
                ]
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    resume_summary = resume_sections.get(
        "summary",
        ""
    )

    jd_summary_parts = [

        jd_sections.get(
            "general",
            ""
        ),

        jd_sections.get(
            "requirements",
            ""
        )
    ]

    jd_summary = " ".join(
        part
        for part in jd_summary_parts
        if part
    ).strip()

    if jd_summary and resume_summary:

        comparisons.append({

            "section":
                "summary",

            "jd":
                jd_summary,

            "resume":
                resume_summary,

            "weight":
                SECTION_WEIGHTS[
                    "summary"
                ]
        })

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    jd_requirements = jd_sections.get(
        "requirements",
        ""
    )

    resume_education = resume_sections.get(
        "education",
        ""
    )

    if jd_requirements and resume_education:

        comparisons.append({

            "section":
                "education",

            "jd":
                jd_requirements,

            "resume":
                resume_education,

            "weight":
                SECTION_WEIGHTS[
                    "education"
                ]
        })

    return comparisons


# ============================================================
# REGISTER UNIQUE EMBEDDING TEXT
# ============================================================

def _register_embedding_text(
    text: str,
    embedding_texts: List[str],
    embedding_index: Dict[str, int]
):
    """
    Register a unique normalized text and return its index.

    Duplicate strings are encoded only once.
    """

    prepared = prepare_text(
        text
    )

    if not prepared:

        return None

    if prepared not in embedding_index:

        embedding_index[
            prepared
        ] = len(
            embedding_texts
        )

        embedding_texts.append(
            prepared
        )

    return embedding_index[
        prepared
    ]


# ============================================================
# BATCH SEMANTIC CALCULATION
# ============================================================

def _calculate_batch_semantic_scores(
    jd_text: str,
    resume_texts: List[str],
    resume_sections_list: List[Dict[str, str]],
    jd_sections: Dict[str, str]
):
    """
    Generate all semantic embeddings needed for the current
    candidate batch using one model operation.

    Returns:
        overall_scores
        section_scores
        sections_compared
    """

    embedding_texts = []

    embedding_index = {}

    # --------------------------------------------------------
    # Overall JD embedding
    # --------------------------------------------------------

    jd_overall_index = _register_embedding_text(

        jd_text,

        embedding_texts,

        embedding_index
    )

    # --------------------------------------------------------
    # Overall resume embeddings
    # --------------------------------------------------------

    resume_overall_indices = []

    for resume_text in resume_texts:

        index = _register_embedding_text(

            resume_text,

            embedding_texts,

            embedding_index
        )

        resume_overall_indices.append(
            index
        )

    # --------------------------------------------------------
    # Section comparison indices
    # --------------------------------------------------------

    candidate_section_data = []

    for resume_sections in resume_sections_list:

        comparisons = _build_section_comparisons(

            jd_sections,

            resume_sections
        )

        section_data = []

        for comparison in comparisons:

            jd_index = _register_embedding_text(

                comparison["jd"],

                embedding_texts,

                embedding_index
            )

            resume_index = _register_embedding_text(

                comparison["resume"],

                embedding_texts,

                embedding_index
            )

            section_data.append({

                "section":
                    comparison["section"],

                "weight":
                    comparison["weight"],

                "jd_index":
                    jd_index,

                "resume_index":
                    resume_index
            })

        candidate_section_data.append(
            section_data
        )

    # --------------------------------------------------------
    # Generate all embeddings in ONE operation
    # --------------------------------------------------------

    if not embedding_texts:

        return (

            [
                0.0
                for _ in resume_texts
            ],

            [
                {}
                for _ in resume_texts
            ],

            [
                []
                for _ in resume_texts
            ]
        )

    embeddings = generate_embeddings(
        embedding_texts
    )

    if (
        embeddings is None
        or embeddings.ndim != 2
        or embeddings.shape[1] == 0
    ):

        return (

            [
                0.0
                for _ in resume_texts
            ],

            [
                {}
                for _ in resume_texts
            ],

            [
                []
                for _ in resume_texts
            ]
        )

    # --------------------------------------------------------
    # Overall scores
    # --------------------------------------------------------

    overall_scores = []

    for resume_index in resume_overall_indices:

        if (
            jd_overall_index is None
            or resume_index is None
        ):

            overall_scores.append(
                0.0
            )

            continue

        score = cosine_score(

            embeddings[
                jd_overall_index
            ],

            embeddings[
                resume_index
            ]
        )

        overall_scores.append(
            round(
                score * 100,
                2
            )
        )

    # --------------------------------------------------------
    # Section scores
    # --------------------------------------------------------

    section_scores_list = []

    section_names_list = []

    for section_data in candidate_section_data:

        section_scores = {}

        weighted_scores = []

        for item in section_data:

            jd_index = item[
                "jd_index"
            ]

            resume_index = item[
                "resume_index"
            ]

            if (
                jd_index is None
                or resume_index is None
            ):

                continue

            score = cosine_score(

                embeddings[
                    jd_index
                ],

                embeddings[
                    resume_index
                ]
            ) * 100

            score = round(
                max(
                    0.0,
                    min(
                        100.0,
                        score
                    )
                ),
                2
            )

            section = item[
                "section"
            ]

            weight = item[
                "weight"
            ]

            section_scores[
                section
            ] = score

            weighted_scores.append(
                (
                    score,
                    weight
                )
            )

        section_scores_list.append(
            section_scores
        )

        section_names_list.append(
            list(
                section_scores.keys()
            )
        )

    # --------------------------------------------------------
    # Weighted section score
    #
    # This keeps the same behavior as semantic_matcher.py:
    # available sections are normalized by their available
    # weights.
    # --------------------------------------------------------

    section_aware_scores = []

    for candidate_index in range(
        len(resume_texts)
    ):

        section_data = candidate_section_data[
            candidate_index
        ]

        if not section_data:

            section_aware_scores.append(
                overall_scores[
                    candidate_index
                ]
            )

            continue

        weighted_scores = []

        for item in section_data:

            section = item[
                "section"
            ]

            if section not in section_scores_list[
                candidate_index
            ]:

                continue

            score = section_scores_list[
                candidate_index
            ][
                section
            ]

            weighted_scores.append(
                (
                    score,
                    item["weight"]
                )
            )

        if weighted_scores:

            numerator = sum(
                score * weight
                for score, weight
                in weighted_scores
            )

            denominator = sum(
                weight
                for _, weight
                in weighted_scores
            )

            if denominator > 0:

                final_score = (
                    numerator /
                    denominator
                )

            else:

                final_score = overall_scores[
                    candidate_index
                ]

        else:

            final_score = overall_scores[
                candidate_index
            ]

        section_aware_scores.append(
            round(
                float(
                    final_score
                ),
                2
            )
        )

    return (
        section_aware_scores,
        section_scores_list,
        section_names_list
    )


# ============================================================
# BATCH RESUME ANALYSIS
# ============================================================

def analyze_resumes_batch(
    jd_text: str,
    resume_data: List[Tuple[str, str]],
    top_n: int = 20
) -> List[Dict]:
    """
    Analyze multiple resumes efficiently.

    Input:
        jd_text:
            Job description.

        resume_data:
            [
                ("resume1.pdf", "resume text"),
                ("resume2.docx", "resume text"),
                ...
            ]

    Output:
        List of analysis dictionaries.

    Performance:
        - JD sections extracted once.
        - JD keywords extracted once.
        - TF-IDF fitted once across all documents.
        - Semantic model called once for the entire batch.
    """

    if not jd_text or not resume_data:

        return []

    # --------------------------------------------------------
    # Keep valid resumes
    # --------------------------------------------------------

    valid_data = []

    for name, text in resume_data:

        if not text:
            continue

        prepared = prepare_text(
            text
        )

        if not prepared:
            continue

        valid_data.append(
            (
                name,
                text
            )
        )

    if not valid_data:

        return []

    resume_names = [
        name
        for name, _ in valid_data
    ]

    resume_texts = [
        text
        for _, text in valid_data
    ]

    # --------------------------------------------------------
    # Extract JD information ONCE
    # --------------------------------------------------------

    jd_keywords = extract_top_keywords(
        jd_text,
        n=top_n
    )

    try:

        jd_sections = extract_jd_sections(
            jd_text
        )

    except Exception:

        jd_sections = {}

    # --------------------------------------------------------
    # Extract resume sections ONCE
    # --------------------------------------------------------

    resume_sections_list = []

    for resume_text in resume_texts:

        try:

            sections = extract_resume_sections(
                resume_text
            )

        except Exception:

            sections = {}

        resume_sections_list.append(
            sections
        )

    # --------------------------------------------------------
    # BATCH lexical similarity
    # --------------------------------------------------------

    lexical_scores = compute_lexical_scores_batch(
        jd_text,
        resume_texts
    )

    # --------------------------------------------------------
    # BATCH semantic similarity
    # --------------------------------------------------------

    (
        semantic_scores,
        section_scores_list,
        sections_compared_list
    ) = _calculate_batch_semantic_scores(

        jd_text,

        resume_texts,

        resume_sections_list,

        jd_sections
    )

    # --------------------------------------------------------
    # Build candidate results
    # --------------------------------------------------------

    results = []

    for index, resume_text in enumerate(
        resume_texts
    ):

        # --------------------------------------------
        # Safe indexing
        # --------------------------------------------

        lexical_score = (
            lexical_scores[index]
            if index < len(lexical_scores)
            else 0.0
        )

        semantic_score = (
            semantic_scores[index]
            if index < len(semantic_scores)
            else 0.0
        )

        section_scores = (

            section_scores_list[index]

            if index < len(
                section_scores_list
            )

            else {}
        )

        sections_compared = (

            sections_compared_list[index]

            if index < len(
                sections_compared_list
            )

            else []
        )

        # --------------------------------------------
        # Keywords
        # --------------------------------------------

        matched_keywords, missing_keywords = (
            find_keyword_matches(
                jd_text,
                resume_text,
                jd_keywords
            )
        )

        # --------------------------------------------
        # Skills
        # --------------------------------------------

        matched_skills, missing_skills = (
            compare_skills(
                jd_text,
                resume_text
            )
        )

        skill_score = calculate_skill_match_score(
            jd_text,
            resume_text
        )

        category_analysis = (
            compare_skills_by_category(
                jd_text,
                resume_text
            )
        )

        skill_statistics = (
            get_skill_statistics(
                jd_text,
                resume_text
            )
        )

        # --------------------------------------------
        # Resume sections
        # --------------------------------------------

        resume_sections = (
            resume_sections_list[index]
        )

        # --------------------------------------------
        # Section-aware result
        # --------------------------------------------

        section_aware_score = semantic_score

        # --------------------------------------------
        # Return same structure as analyze_resume()
        # --------------------------------------------

        results.append({

            "resume_name":
                resume_names[index],

            "lexical_score":
                lexical_score,

            "semantic_score":
                semantic_score,

            "section_aware_score":
                section_aware_score,

            "section_scores":
                section_scores,

            "sections_compared":
                sections_compared,

            "jd_sections":
                list(
                    jd_sections.keys()
                ),

            "resume_sections":
                list(
                    resume_sections.keys()
                ),

            "keywords":
                jd_keywords,

            "matched_keywords":
                matched_keywords,

            "missing_keywords":
                missing_keywords,

            "matched_skills":
                matched_skills,

            "missing_skills":
                missing_skills,

            "skill_score":
                skill_score,

            "category_analysis":
                category_analysis,

            "skill_statistics":
                skill_statistics
        })

    return results


# ============================================================
# SINGLE RESUME ANALYSIS
# ============================================================

def analyze_resume(
    jd_text: str,
    resume_text: str,
    top_n: int = 20
) -> Dict:
    """
    Backward-compatible single resume analysis.

    Internally uses the batch engine so app.py and other
    modules can continue using the old function name.
    """

    results = analyze_resumes_batch(

        jd_text,

        [
            (
                "resume",
                resume_text
            )
        ],

        top_n=top_n
    )

    if not results:

        return {

            "lexical_score": 0.0,

            "semantic_score": 0.0,

            "section_aware_score": 0.0,

            "section_scores": {},

            "sections_compared": [],

            "jd_sections": [],

            "resume_sections": [],

            "keywords": [],

            "matched_keywords": [],

            "missing_keywords": [],

            "matched_skills": [],

            "missing_skills": [],

            "skill_score": 0.0,

            "category_analysis": {},

            "skill_statistics": {}
        }

    return results[0]


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    jd = """
    We are looking for an AI Engineer with strong Python,
    machine learning, deep learning and NLP experience.
    The candidate should have experience with PyTorch,
    TensorFlow, FastAPI, Docker and SQL.
    """

    resume_1 = """
    AI undergraduate with experience in Python,
    machine learning, NLP and deep learning.

    TECHNICAL SKILLS
    Python, PyTorch, TensorFlow, SQL, Docker, Git.

    PROJECTS
    Built an AI Resume Analyzer using NLP.

    EXPERIENCE
    Developed machine learning applications.
    """

    resume_2 = """
    Software developer with Java, SQL and Spring Boot
    experience.

    TECHNICAL SKILLS
    Java, SQL, React, Git.

    PROJECTS
    Developed enterprise web applications.

    EXPERIENCE
    Backend development experience.
    """

    resume_data = [

        (
            "resume1.docx",
            resume_1
        ),

        (
            "resume2.docx",
            resume_2
        )
    ]

    print("=" * 70)
    print("BATCH ANALYSIS ENGINE TEST")
    print("=" * 70)

    results = analyze_resumes_batch(
        jd,
        resume_data,
        top_n=20
    )

    for result in results:

        print("\n" + "-" * 70)

        print(
            "Resume:",
            result["resume_name"]
        )

        print(
            "Lexical:",
            result["lexical_score"]
        )

        print(
            "Semantic:",
            result["semantic_score"]
        )

        print(
            "Skill:",
            result["skill_score"]
        )

        print(
            "Matched Skills:",
            result["matched_skills"]
        )

        print(
            "Missing Skills:",
            result["missing_skills"]
        )

        print(
            "Matched Keywords:",
            result["matched_keywords"]
        )

        print(
            "Missing Keywords:",
            result["missing_keywords"]
        )

        print(
            "Section Scores:",
            result["section_scores"]
        )

    print("\n" + "=" * 70)
    print("BATCH TEST COMPLETE")
    print("=" * 70)