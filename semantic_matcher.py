"""
AI Resume Screening Platform
============================

Semantic Matching Engine

Features:
    - Cached Sentence Transformer model
    - Batch embedding generation
    - Whole-document semantic similarity
    - Section-aware semantic similarity
    - Candidate semantic ranking
    - Safe empty-input handling
    - Faster repeated Streamlit analysis

Model:
    all-MiniLM-L6-v2

Important:
    Semantic similarity is one signal in the project-defined
    ATS scoring system. It is not an official ATS formula.
"""

from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

try:
    from streamlit import cache_resource
except ModuleNotFoundError:
    def cache_resource(func=None, **kwargs):
        def decorator(target):
            return lru_cache(maxsize=1)(target)

        if func is not None:
            return decorator(func)

        return decorator



# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# Limit raw text size before sending it to the model.
# This reduces unnecessary processing on very large resumes.
MAX_CHARACTERS = 10000


# ============================================================
# CACHED MODEL
# ============================================================

@cache_resource(
    show_spinner="Loading semantic AI model..."
)
def get_model() -> SentenceTransformer:
    """
    Load the Sentence Transformer once and reuse it.

    Streamlit cache_resource keeps the model alive across
    reruns instead of loading it repeatedly.
    """

    return SentenceTransformer(
        MODEL_NAME
    )


# ============================================================
# TEXT PREPARATION
# ============================================================

def prepare_text(
    text: str,
    max_characters: int = MAX_CHARACTERS
) -> str:
    """
    Normalize text before embedding.
    """

    if not text:
        return ""

    text = str(
        text
    )

    # Normalize whitespace
    text = " ".join(
        text.split()
    )

    if len(text) > max_characters:

        text = text[
            :max_characters
        ]

    return text.strip()


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embedding(
    text: str
):
    """
    Generate one normalized embedding.

    Returns:
        numpy array
        or None for empty input.
    """

    prepared = prepare_text(
        text
    )

    if not prepared:
        return None

    model = get_model()

    embedding = model.encode(
        prepared,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding


def generate_embeddings(
    texts: List[str]
):
    """
    Generate embeddings for multiple texts in one batch.

    Empty texts are represented internally by zero vectors
    so the output remains aligned with the input list.

    Returns:
        numpy.ndarray
    """

    if not texts:

        return np.empty(
            (0, 0)
        )

    prepared_texts = [
        prepare_text(text)
        for text in texts
    ]

    valid_indices = [
        index
        for index, text in enumerate(
            prepared_texts
        )
        if text
    ]

    if not valid_indices:

        return np.empty(
            (len(texts), 0)
        )

    valid_texts = [
        prepared_texts[index]
        for index in valid_indices
    ]

    model = get_model()

    valid_embeddings = model.encode(
        valid_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16
    )

    valid_embeddings = np.asarray(
        valid_embeddings,
        dtype=np.float32
    )

    # Create output aligned with original input.
    embedding_dimension = valid_embeddings.shape[1]

    output = np.zeros(
        (
            len(texts),
            embedding_dimension
        ),
        dtype=np.float32
    )

    for output_index, original_index in enumerate(
        valid_indices
    ):

        output[original_index] = (
            valid_embeddings[output_index]
        )

    return output


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_score(
    vector_a,
    vector_b
) -> float:
    """
    Compute cosine similarity between two normalized vectors.

    Because embeddings are normalized, cosine similarity is
    equivalent to their dot product.
    """

    if vector_a is None or vector_b is None:

        return 0.0

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32
    )

    if vector_a.size == 0 or vector_b.size == 0:

        return 0.0

    if vector_a.shape != vector_b.shape:

        return 0.0

    score = float(
        np.dot(
            vector_a,
            vector_b
        )
    )

    # Numerical safety
    score = max(
        0.0,
        min(
            1.0,
            score
        )
    )

    return score


# ============================================================
# WHOLE DOCUMENT SIMILARITY
# ============================================================

def compute_semantic_similarity(
    jd_text: str,
    resume_text: str
) -> float:
    """
    Calculate semantic similarity between a JD and a resume.

    Returns:
        Score from 0 to 100.
    """

    jd_clean = prepare_text(
        jd_text
    )

    resume_clean = prepare_text(
        resume_text
    )

    if not jd_clean or not resume_clean:

        return 0.0

    embeddings = generate_embeddings(
        [
            jd_clean,
            resume_clean
        ]
    )

    if embeddings.shape[1] == 0:

        return 0.0

    score = cosine_score(
        embeddings[0],
        embeddings[1]
    )

    return round(
        score * 100,
        2
    )


# ============================================================
# SECTION SIMILARITY
# ============================================================

def compute_section_similarity(
    jd_section: str,
    resume_section: str
) -> float:
    """
    Compute semantic similarity between two sections.
    """

    if not jd_section or not resume_section:

        return 0.0

    return compute_semantic_similarity(
        jd_section,
        resume_section
    )


# ============================================================
# SECTION WEIGHTS
# ============================================================

SECTION_WEIGHTS = {
    "skills": 0.35,
    "experience": 0.30,
    "projects": 0.20,
    "summary": 0.10,
    "education": 0.05
}


# ============================================================
# SECTION-AWARE SEMANTIC SIMILARITY
# ============================================================

def compute_section_aware_similarity(
    jd_sections: Dict[str, str],
    resume_sections: Dict[str, str],
    fallback_jd: str = "",
    fallback_resume: str = ""
) -> Dict:
    """
    Calculate weighted semantic similarity across resume
    sections.

    Section embeddings are generated in a single batch,
    which is faster than independently encoding every pair.
    """

    if not jd_sections:

        jd_sections = {}

    if not resume_sections:

        resume_sections = {}

    # --------------------------------------------------------
    # Build section comparison pairs
    # --------------------------------------------------------

    comparisons = []

    # Skills
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
            "section": "skills",
            "jd": jd_skills,
            "resume": resume_skills,
            "weight": SECTION_WEIGHTS[
                "skills"
            ]
        })

    # Experience
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
            "section": "experience",
            "jd": jd_experience,
            "resume": resume_experience,
            "weight": SECTION_WEIGHTS[
                "experience"
            ]
        })

    # Projects
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
            "section": "projects",
            "jd": jd_projects,
            "resume": resume_projects,
            "weight": SECTION_WEIGHTS[
                "projects"
            ]
        })

    # Summary
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
            "section": "summary",
            "jd": jd_summary,
            "resume": resume_summary,
            "weight": SECTION_WEIGHTS[
                "summary"
            ]
        })

    # Education
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
            "section": "education",
            "jd": jd_requirements,
            "resume": resume_education,
            "weight": SECTION_WEIGHTS[
                "education"
            ]
        })

    # --------------------------------------------------------
    # No comparable sections
    # --------------------------------------------------------

    if not comparisons:

        return {
            "section_scores": {},
            "section_aware_score":
                compute_semantic_similarity(
                    fallback_jd,
                    fallback_resume
                ),
            "sections_compared": []
        }

    # --------------------------------------------------------
    # BATCH EMBEDDINGS
    # --------------------------------------------------------

    texts = []

    for comparison in comparisons:

        texts.append(
            comparison["jd"]
        )

        texts.append(
            comparison["resume"]
        )

    embeddings = generate_embeddings(
        texts
    )

    if embeddings.shape[1] == 0:

        return {
            "section_scores": {},
            "section_aware_score": 0.0,
            "sections_compared": []
        }

    # --------------------------------------------------------
    # Calculate section scores
    # --------------------------------------------------------

    section_scores = {}
    weighted_scores = []

    for index, comparison in enumerate(
        comparisons
    ):

        jd_embedding = embeddings[
            index * 2
        ]

        resume_embedding = embeddings[
            index * 2 + 1
        ]

        score = cosine_score(
            jd_embedding,
            resume_embedding
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

        section = comparison[
            "section"
        ]

        weight = comparison[
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

    # --------------------------------------------------------
    # Normalize available weights
    # --------------------------------------------------------

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

        final_score = 0.0

    return {
        "section_scores":
            section_scores,

        "section_aware_score":
            round(
                float(final_score),
                2
            ),

        "sections_compared":
            list(
                section_scores.keys()
            )
    }


# ============================================================
# SEMANTIC RESUME RANKING
# ============================================================

def rank_resumes_semantically(
    jd_text: str,
    resume_data: List[Tuple[str, str]]
) -> List[Dict]:
    """
    Rank multiple resumes using one batched embedding call.

    Input:
        jd_text
        [
            ("resume1.pdf", text1),
            ("resume2.pdf", text2),
            ...
        ]

    Output:
        [
            {
                "resume_name": "...",
                "semantic_score": 82.5
            }
        ]
    """

    if not jd_text or not resume_data:

        return []

    valid_data = []

    for name, text in resume_data:

        cleaned = prepare_text(
            text
        )

        if cleaned:

            valid_data.append(
                (
                    name,
                    cleaned
                )
            )

    if not valid_data:

        return []

    # --------------------------------------------------------
    # Batch JD + all resumes
    # --------------------------------------------------------

    texts = [
        prepare_text(jd_text)
    ]

    texts.extend(
        text
        for _, text in valid_data
    )

    embeddings = generate_embeddings(
        texts
    )

    if embeddings.shape[1] == 0:

        return []

    jd_embedding = embeddings[
        0
    ]

    resume_embeddings = embeddings[
        1:
    ]

    results = []

    for index, (
        name,
        _
    ) in enumerate(valid_data):

        score = cosine_score(
            jd_embedding,
            resume_embeddings[
                index
            ]
        )

        results.append({
            "resume_name":
                name,

            "semantic_score":
                round(
                    score * 100,
                    2
                )
        })

    results.sort(
        key=lambda item:
            item["semantic_score"],
        reverse=True
    )

    return results


# ============================================================
# SCORE INTERPRETATION
# ============================================================

def interpret_semantic_score(
    score: float
) -> str:
    """
    Human-readable interpretation.
    """

    score = float(
        score
    )

    if score >= 80:

        return (
            "High semantic alignment."
        )

    if score >= 60:

        return (
            "Moderate semantic alignment."
        )

    if score >= 40:

        return (
            "Some semantic alignment detected."
        )

    return (
        "Low semantic alignment."
    )


# ============================================================
# COMPLETE SEMANTIC ANALYSIS
# ============================================================

def analyze_semantic_match(
    jd_text: str,
    resume_text: str
) -> Dict:
    """
    Complete semantic analysis.
    """

    score = compute_semantic_similarity(
        jd_text,
        resume_text
    )

    return {

        "semantic_score":
            score,

        "interpretation":
            interpret_semantic_score(
                score
            ),

        "model":
            MODEL_NAME
    }


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_semantic_model_cache():
    """
    Clear the cached Sentence Transformer model.

    Useful during development if you change the model name
    or need to force a fresh model load.
    """

    try:

        get_model.clear()

    except Exception:

        pass


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    jd = """
    We are looking for an AI Engineer with strong Python,
    machine learning, deep learning and NLP experience.
    The candidate should have experience building and
    deploying machine learning applications using PyTorch,
    TensorFlow, FastAPI and Docker.
    """

    resume_1 = """
    AI undergraduate with experience in Python, machine
    learning, deep learning and NLP. Built applications
    using PyTorch, TensorFlow, FastAPI and Docker.
    """

    resume_2 = """
    Software developer with Java, Spring Boot and SQL
    experience. Worked primarily on enterprise web
    applications and backend systems.
    """

    print("=" * 65)
    print("SEMANTIC MATCHING ENGINE TEST")
    print("=" * 65)

    print("\nSingle Resume Test")
    print("-" * 65)

    score = compute_semantic_similarity(
        jd,
        resume_1
    )

    print(
        "Semantic Score:",
        score
    )

    print(
        "Interpretation:",
        interpret_semantic_score(
            score
        )
    )

    print("\nBatch Ranking Test")
    print("-" * 65)

    ranking = rank_resumes_semantically(
        jd,
        [
            (
                "resume1.docx",
                resume_1
            ),
            (
                "resume2.docx",
                resume_2
            )
        ]
    )

    for item in ranking:

        print(
            item
        )

    print("\nSection-Aware Test")
    print("-" * 65)

    jd_sections = {
        "skills": """
        Python, machine learning, deep learning,
        NLP, PyTorch and Docker
        """,

        "requirements": """
        Experience building machine learning systems
        and deploying applications.
        """
    }

    resume_sections = {
        "skills": """
        Python, machine learning, NLP,
        PyTorch and TensorFlow
        """,

        "experience": """
        Developed and deployed machine learning
        applications.
        """
    }

    section_result = (
        compute_section_aware_similarity(
            jd_sections,
            resume_sections,
            fallback_jd=jd,
            fallback_resume=resume_1
        )
    )

    print(
        section_result
    )

    print("\n" + "=" * 65)
    print("TEST COMPLETE")
    print("=" * 65)
