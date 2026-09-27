"""
RAG Recruiter Copilot

A lightweight, portfolio-friendly RAG pipeline for recruiter questions.

Retrieval is local:
    TF-IDF cosine similarity + keyword overlap

Evidence sources:
    - Job Description
    - Resume
    - Candidate analysis
    - Requirement matching
    - Resume intelligence
    - Resume quality
    - Screening brief
    - Interview history
    - Communication history

Generation is optional:
    Gemini API using only the retrieved evidence.

The deterministic ATS and requirement scores are never recomputed.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_GEMINI_MODEL = os.getenv(
    "GEMINI_COPILOT_MODEL",
    "gemini-3.5-flash-lite",
)

GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

MAX_CHUNK_CHARS = 1400
CHUNK_OVERLAP = 180
MAX_CONTEXT_CHARS = 9000


def clean_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    )


def token_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9+#.\-]{1,}",
            str(text or "").lower(),
        )
    )


def keyword_overlap(query: str, text: str) -> float:
    q = token_set(query)
    d = token_set(text)

    if not q or not d:
        return 0.0

    return len(q & d) / len(q)


def chunk_text(
    text: str,
    *,
    source: str,
    section: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split a document into overlapping evidence chunks.
    """
    normalized = str(text or "").strip()

    if not normalized:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n|\r\n\s*\r\n",
            normalized,
        )
        if p.strip()
    ]

    if not paragraphs:
        paragraphs = [normalized]

    chunks: List[Dict[str, Any]] = []
    current = ""

    def add(value: str) -> None:
        value = value.strip()
        if value:
            chunks.append(
                {
                    "source": source,
                    "section": section,
                    "text": value,
                }
            )

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            if not current:
                current = paragraph
            elif (
                len(current) + len(paragraph) + 1
                <= max_chars
            ):
                current += " " + paragraph
            else:
                add(current)
                tail = (
                    current[-overlap:]
                    if overlap > 0
                    else ""
                )
                current = (
                    f"{tail} {paragraph}".strip()
                    if tail
                    else paragraph
                )
        else:
            if current:
                add(current)
                current = ""

            start = 0

            while start < len(paragraph):
                end = min(
                    start + max_chars,
                    len(paragraph),
                )

                add(
                    paragraph[start:end]
                )

                if end >= len(paragraph):
                    break

                start = max(
                    end - overlap,
                    start + 1,
                )

    if current:
        add(current)

    for index, item in enumerate(
        chunks,
        start=1,
    ):
        item["id"] = index

    return chunks


def _add_items(
    chunks: List[Dict[str, Any]],
    values: Iterable[Any],
    *,
    source: str,
    section: str,
) -> None:
    cleaned = []

    for value in values:
        text = clean_text(value)
        if text:
            cleaned.append(text)

    if cleaned:
        chunks.extend(
            chunk_text(
                "\n".join(cleaned),
                source=source,
                section=section,
            )
        )


def build_copilot_corpus(
    *,
    candidate_row: Dict[str, Any],
    resume_text: str,
    jd_text: str,
    analysis_result: Dict[str, Any],
    intelligence: Dict[str, Any],
    quality: Dict[str, Any],
    requirements: Dict[str, Any],
    screening_brief: Optional[Dict[str, Any]] = None,
    interview_history: Optional[Sequence[Dict[str, Any]]] = None,
    communication_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build one candidate-specific RAG corpus.
    """
    chunks: List[Dict[str, Any]] = []

    # Job Description
    chunks.extend(
        chunk_text(
            jd_text,
            source="Job Description",
            section="Full JD",
        )
    )

    # Resume
    chunks.extend(
        chunk_text(
            resume_text,
            source="Resume",
            section="Full Resume",
        )
    )

    # Matching signals
    _add_items(
        chunks,
        analysis_result.get(
            "matched_skills",
            [],
        ) or [],
        source="Candidate Analysis",
        section="Matched Skills",
    )

    _add_items(
        chunks,
        analysis_result.get(
            "missing_skills",
            [],
        ) or [],
        source="Candidate Analysis",
        section="Missing Skills",
    )

    _add_items(
        chunks,
        analysis_result.get(
            "matched_keywords",
            [],
        ) or [],
        source="Candidate Analysis",
        section="Matched Keywords",
    )

    section_scores = analysis_result.get(
        "section_scores",
        {},
    ) or {}

    if section_scores:
        _add_items(
            chunks,
            [
                f"{key}: {value}"
                for key, value
                in section_scores.items()
            ],
            source="Candidate Analysis",
            section="Section Scores",
        )

    # Requirements
    required = requirements.get(
        "required_skills",
        {},
    ) or {}

    preferred = requirements.get(
        "preferred_skills",
        {},
    ) or {}

    experience = requirements.get(
        "experience",
        {},
    ) or {}

    education = requirements.get(
        "education",
        {},
    ) or {}

    requirement_items = [
        (
            "Requirement Match Score: "
            + str(
                requirements.get(
                    "requirement_match_score"
                )
            )
        ),
        (
            "Required skills matched: "
            + ", ".join(
                map(
                    str,
                    required.get(
                        "matched",
                        [],
                    ) or [],
                )
            )
        ),
        (
            "Required skills not detected: "
            + ", ".join(
                map(
                    str,
                    required.get(
                        "missing",
                        [],
                    ) or [],
                )
            )
        ),
        (
            "Preferred skills matched: "
            + ", ".join(
                map(
                    str,
                    preferred.get(
                        "matched",
                        [],
                    ) or [],
                )
            )
        ),
        (
            "Preferred skills not detected: "
            + ", ".join(
                map(
                    str,
                    preferred.get(
                        "missing",
                        [],
                    ) or [],
                )
            )
        ),
        (
            "Experience requirement status: "
            + str(
                experience.get(
                    "status",
                    "Not specified",
                )
            )
        ),
        (
            "Candidate experience used for matching: "
            + str(
                experience.get(
                    "candidate_years",
                    intelligence.get(
                        "experience_years",
                        0,
                    ),
                )
            )
        ),
        (
            "Required experience: "
            + str(
                experience.get(
                    "required_years",
                    0,
                )
            )
        ),
        (
            "Education requirement status: "
            + str(
                education.get(
                    "status",
                    "Not specified",
                )
            ),
        ),
    ]

    _add_items(
        chunks,
        requirement_items,
        source="Requirement Matching",
        section="Role Requirements",
    )

    # Resume intelligence
    intelligence_items = [
        (
            "Candidate: "
            + str(
                candidate_row.get(
                    "Candidate",
                    "Candidate",
                )
            )
        ),
        (
            "Experience years: "
            + str(
                intelligence.get(
                    "experience_years",
                    0,
                )
            )
        ),
        (
            "Education: "
            + clean_text(
                intelligence.get(
                    "education"
                )
            )
        ),
        (
            "Projects: "
            + clean_text(
                intelligence.get(
                    "projects"
                )
            )
        ),
        (
            "Certifications: "
            + clean_text(
                intelligence.get(
                    "certifications"
                )
            )
        ),
        (
            "Detected skills: "
            + ", ".join(
                map(
                    str,
                    intelligence.get(
                        "all_skills",
                        [],
                    ) or [],
                )
            )
        ),
        (
            "Section completeness: "
            + str(
                intelligence.get(
                    "section_completeness",
                    0,
                )
            )
            + "%"
        ),
        (
            "Contact completeness: "
            + str(
                intelligence.get(
                    "contact_completeness",
                    0,
                )
            )
            + "%"
        ),
    ]

    _add_items(
        chunks,
        intelligence_items,
        source="Resume Intelligence",
        section="Candidate Facts",
    )

    # Resume quality
    quality_items = [
        (
            "Resume Quality Score: "
            + str(
                quality.get(
                    "quality_score",
                    0,
                )
            )
            + "%"
        ),
        (
            "Action Verb Score: "
            + str(
                (
                    quality.get(
                        "action_verbs",
                        {},
                    )
                    or {}
                ).get(
                    "score",
                    0,
                )
            )
            + "%"
        ),
        (
            "Quantification Score: "
            + str(
                (
                    quality.get(
                        "quantification",
                        {},
                    )
                    or {}
                ).get(
                    "score",
                    0,
                )
            )
            + "%"
        ),
        (
            "Repetition Score: "
            + str(
                (
                    quality.get(
                        "repetition",
                        {},
                    )
                    or {}
                ).get(
                    "score",
                    0,
                )
            )
            + "%"
        ),
    ]

    quality_items.extend(
        quality.get(
            "recommendations",
            [],
        ) or []
    )

    _add_items(
        chunks,
        quality_items,
        source="Resume Quality",
        section="Quality Signals",
    )

    # Screening Brief
    if screening_brief:
        for key, label in [
            ("summary", "Summary"),
            ("strengths", "Strengths"),
            ("gaps", "Gaps"),
            ("interview_focus", "Interview Focus"),
            ("talking_points", "Recruiter Talking Points"),
            ("interview_questions", "Suggested Questions"),
        ]:
            value = screening_brief.get(
                key
            )

            if isinstance(value, list):
                values = [
                    f"{label}: {item}"
                    for item in value
                ]
            elif value:
                values = [
                    f"{label}: {value}"
                ]
            else:
                values = []

            _add_items(
                chunks,
                values,
                source="Screening Brief",
                section=label,
            )

    # Interview history
    for interview in interview_history or []:
        interview_items = [
            f"Interview Round: {interview.get('interview_round', '')}",
            f"Interview Type: {interview.get('interview_type', '')}",
            f"Interview Date: {interview.get('scheduled_date', '')}",
            f"Interview Time: {interview.get('scheduled_time', '')}",
            f"Interviewer: {interview.get('interviewer', '')}",
            f"Status: {interview.get('status', '')}",
            f"Technical Score: {interview.get('technical_score')}",
            f"Communication Score: {interview.get('communication_score')}",
            f"Problem Solving Score: {interview.get('problem_solving_score')}",
            f"Overall Interview Score: {interview.get('overall_score')}",
            f"Recommendation: {interview.get('recommendation')}",
            f"Feedback: {interview.get('feedback', '')}",
        ]

        _add_items(
            chunks,
            interview_items,
            source="Interview History",
            section=f"Interview #{interview.get('id', '')}",
        )

    # Communication history
    for message in communication_history or []:
        message_items = [
            f"Communication Type: {message.get('communication_type', '')}",
            f"Subject: {message.get('subject', '')}",
            f"Created: {message.get('created_at', '')}",
            f"Body: {message.get('body', '')}",
        ]

        _add_items(
            chunks,
            message_items,
            source="Communication History",
            section=f"Communication #{message.get('id', '')}",
        )

    for index, item in enumerate(
        chunks,
        start=1,
    ):
        item["id"] = index

    return chunks


def retrieve_evidence(
    query: str,
    corpus: Sequence[Dict[str, Any]],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval using TF-IDF similarity plus query-token overlap.
    """
    query = clean_text(query)

    if not query or not corpus:
        return []

    documents = [
        clean_text(
            " ".join(
                [
                    str(item.get("source", "")),
                    str(item.get("section", "")),
                    str(item.get("text", "")),
                ]
            )
        )
        for item in corpus
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=5000,
    )

    try:
        matrix = vectorizer.fit_transform(
            documents + [query]
        )
    except ValueError:
        return []

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1],
    )[0]

    scored = []

    for index, item in enumerate(corpus):
        overlap = keyword_overlap(
            query,
            item.get("text", ""),
        )

        score = (
            0.75 * float(
                similarities[index]
            )
            + 0.25 * float(overlap)
        )

        result = dict(item)
        result["retrieval_score"] = round(
            score,
            4,
        )
        result["tfidf_score"] = round(
            float(similarities[index]),
            4,
        )
        result["keyword_overlap"] = round(
            overlap,
            4,
        )

        scored.append(result)

    scored.sort(
        key=lambda x: x["retrieval_score"],
        reverse=True,
    )

    return scored[:max(1, int(top_k))]


def build_rag_context(
    retrieved: Sequence[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    parts = []
    total_chars = 0

    for index, item in enumerate(
        retrieved,
        start=1,
    ):
        block = (
            f"[SOURCE {index}]\n"
            f"Source: {item.get('source', '')}\n"
            f"Section: {item.get('section', '')}\n"
            f"Evidence: {item.get('text', '')}\n"
        )

        if (
            total_chars + len(block)
            > max_chars
        ):
            break

        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)


def _retryable(error: Exception) -> bool:
    text = str(error).lower()

    return any(
        term in text
        for term in [
            "503",
            "unavailable",
            "high demand",
            "temporarily",
            "429",
            "resource_exhausted",
        ]
    )


def generate_gemini_copilot_answer(
    *,
    candidate_name: str,
    question: str,
    retrieved: Sequence[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an answer grounded only in retrieved evidence.
    """

    api_key = (
        api_key
        if api_key is not None
        else os.getenv(
            "GEMINI_API_KEY",
            "",
        )
    ).strip()

    requested_model = (
        model
        or os.getenv(
            "GEMINI_COPILOT_MODEL",
            DEFAULT_GEMINI_MODEL,
        )
    ).strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    if not retrieved:
        raise RuntimeError(
            "No evidence was retrieved."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. "
            "Run: pip install -U google-genai"
        ) from exc

    context = build_rag_context(
        retrieved
    )

    prompt = f"""
You are a recruiter copilot.

Candidate:
{candidate_name}

Recruiter question:
{question}

Retrieved evidence:
{context}

Answer ONLY from the retrieved evidence.

Rules:
- Do not invent candidate facts.
- Do not assume an undetected skill is absent.
- Do not change or recompute ATS, requirement, semantic, skill,
  lexical, keyword, or resume-quality scores.
- Do not make an automatic hiring decision.
- Clearly say when the evidence is insufficient.
- Mention source names when useful.
- Suggested interview questions should validate evidence or uncertainty.
- Keep the answer practical and concise.
""".strip()

    client = genai.Client(
        api_key=api_key
    )

    models = [
        requested_model,
        *[
            candidate
            for candidate
            in GEMINI_FALLBACK_MODELS
            if candidate != requested_model
        ],
    ]

    response = None
    used_model = requested_model
    last_error: Optional[Exception] = None

    for candidate_model in models:
        try:
            response = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
            )

            used_model = candidate_model
            break

        except Exception as exc:
            last_error = exc

            if not _retryable(exc):
                raise RuntimeError(
                    f"Gemini API request failed: {exc}"
                ) from exc

    if response is None:
        raise RuntimeError(
            "Gemini models were temporarily unavailable. "
            f"Last error: {last_error}"
        )

    answer = str(
        getattr(
            response,
            "text",
            "",
        )
        or ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return {
        "answer": answer,
        "candidate": candidate_name,
        "question": question,
        "model": used_model,
        "source": "Google Gemini Developer API",
        "grounded": True,
        "fallback_used": (
            used_model != requested_model
        ),
    }


def evidence_only_fallback(
    *,
    candidate_name: str,
    retrieved: Sequence[Dict[str, Any]],
) -> str:
    """
    Useful no-API fallback. Never fabricates an answer.
    """
    if not retrieved:
        return (
            f"No local evidence was retrieved for {candidate_name}. "
            "Try a more specific question."
        )

    lines = [
        f"Gemini is unavailable. Here is the retrieved evidence for "
        f"{candidate_name}:"
    ]

    for item in retrieved[:4]:
        lines.append(
            f"- [{item.get('source')} / "
            f"{item.get('section')}] "
            f"{clean_text(item.get('text'))}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    sample = build_copilot_corpus(
        candidate_row={
            "Candidate": "Adhyan Sharma",
            "Resume": "Adhyan.pdf",
        },
        resume_text=(
            "Python machine learning NLP. "
            "Built a Streamlit resume analyzer."
        ),
        jd_text=(
            "Need Python, machine learning and SQL. "
            "NLP preferred."
        ),
        analysis_result={
            "matched_skills": [
                "Python",
                "Machine Learning",
            ],
            "missing_skills": ["SQL"],
            "matched_keywords": ["NLP"],
        },
        intelligence={
            "experience_years": 1.5,
            "all_skills": [
                "Python",
                "Machine Learning",
                "NLP",
            ],
            "projects": "Resume Analyzer",
            "section_completeness": 90,
            "contact_completeness": 100,
        },
        quality={
            "quality_score": 88,
        },
        requirements={
            "requirement_match_score": 78,
            "required_skills": {
                "matched": ["Python"],
                "missing": ["SQL"],
            },
        },
    )

    retrieved = retrieve_evidence(
        "What should I ask about the candidate's machine learning project?",
        sample,
        top_k=5,
    )

    assert retrieved
    assert retrieved[0]["retrieval_score"] >= 0

    context = build_rag_context(
        retrieved
    )

    assert "Resume" in context or "Job Description" in context

    print(
        "RAG recruiter copilot retrieval engine is working correctly."
    )
