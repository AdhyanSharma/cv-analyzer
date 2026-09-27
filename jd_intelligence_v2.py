"""
AI Resume Screening Platform
Hybrid Job Description Intelligence V2

Features:
    - English stop-word removal
    - Recruitment-specific stop-word removal
    - Sentence-level TF-IDF
    - 1-3 gram phrase extraction
    - Technical skill dictionary + aliases
    - Skill normalization
    - Required vs preferred classification
    - Experience requirement extraction
    - Job-title extraction
    - Local retrieval ("RAG-lite") over JD chunks
    - Optional Gemini enrichment using retrieved JD evidence

Design:
    Deterministic extraction is the default and does not require an API.
    Gemini enrichment is optional and grounded only in retrieved JD text.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# ------------------------------------------------------------------
# Generic + recruitment stop words
# ------------------------------------------------------------------

GENERIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "in", "into", "is", "it", "its", "of", "on",
    "or", "our", "that", "the", "their", "this", "to", "under",
    "using", "via", "with", "within", "without", "you", "your",
    "will", "can", "may", "must", "should",
}

RECRUITMENT_STOPWORDS = {
    "candidate", "candidates", "company", "companies", "role",
    "roles", "position", "positions", "job", "jobs", "team",
    "teams", "responsibility", "responsibilities", "responsible",
    "requirement", "requirements", "qualification", "qualifications",
    "experience", "experiences", "skills", "skill", "ability",
    "abilities", "looking", "seeking", "ideal", "applicant",
    "applications", "application", "work", "working", "works",
    "environment", "years", "year", "plus", "preferred", "required",
    "including", "etc", "etcetera", "strong", "good", "excellent",
    "proficient", "proficiency", "knowledge", "understanding",
    "familiarity", "opportunity", "success", "successful",
    "description", "overview", "details", "duties", "qualification",
    "developer", "developers", "engineer", "engineers", "analyst",
    "analysts", "scientist", "scientists", "designer", "designers",
    "architect", "architects", "manager", "managers", "database",
    "databases", "website", "websites", "web", "application",
    "applications", "service", "services", "system", "systems",
}

ALL_STOPWORDS = GENERIC_STOPWORDS | RECRUITMENT_STOPWORDS


# ------------------------------------------------------------------
# Canonical technical skills and aliases
# ------------------------------------------------------------------

SKILL_ALIASES: Dict[str, set[str]] = {
    "Python": {"python"},
    "Java": {"java"},
    "JavaScript": {"javascript", "java script", "js"},
    "TypeScript": {"typescript", "ts"},
    "C++": {"c++", "cpp", "cplusplus"},
    "C#": {"c#", "csharp", "c sharp"},
    "SQL": {"sql"},
    "HTML": {"html"},
    "CSS": {"css"},
    "React": {"react", "react.js", "reactjs"},
    "Angular": {"angular", "angular.js", "angularjs"},
    "Vue.js": {"vue", "vue.js", "vuejs"},
    "Node.js": {"node", "node.js", "nodejs"},
    "Django": {"django"},
    "Flask": {"flask"},
    "FastAPI": {"fastapi"},
    "Spring": {"spring", "spring boot"},
    "REST API": {"rest api", "restful api", "rest apis", "restful apis"},
    "GraphQL": {"graphql"},
    "MySQL": {"mysql"},
    "PostgreSQL": {"postgresql", "postgres"},
    "Oracle": {"oracle"},
    "MongoDB": {"mongodb", "mongo db", "mongo"},
    "Redis": {"redis"},
    "SQLite": {"sqlite"},
    "Firebase": {"firebase"},
    "AWS": {"aws", "amazon web services"},
    "Azure": {"azure", "microsoft azure"},
    "Google Cloud": {"gcp", "google cloud", "google cloud platform"},
    "Docker": {"docker"},
    "Kubernetes": {"kubernetes", "k8s"},
    "Git": {"git"},
    "GitHub": {"github"},
    "Jenkins": {"jenkins"},
    "CI/CD": {"ci/cd", "cicd", "continuous integration", "continuous delivery"},
    "Linux": {"linux"},
    "TensorFlow": {"tensorflow"},
    "PyTorch": {"pytorch"},
    "Keras": {"keras"},
    "Scikit-learn": {"scikit-learn", "scikit learn", "sklearn"},
    "Pandas": {"pandas"},
    "NumPy": {"numpy"},
    "OpenCV": {"opencv", "open cv"},
    "Machine Learning": {"machine learning", "ml"},
    "Deep Learning": {"deep learning"},
    "Natural Language Processing": {"natural language processing", "nlp"},
    "Computer Vision": {"computer vision"},
    "Generative AI": {"generative ai", "genai", "gen ai"},
    "Large Language Models": {"large language model", "large language models", "llm", "llms"},
    "RAG": {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
    "Transformers": {"transformers", "transformer models"},
    "Vision Transformer": {"vision transformer", "vit"},
    "GAN": {"gan", "generative adversarial network", "generative adversarial networks"},
    "NLP": {"nlp"},
    "Data Science": {"data science"},
    "Data Analysis": {"data analysis", "data analytics"},
    "Power BI": {"power bi"},
    "Tableau": {"tableau"},
    "Excel": {"microsoft excel", "excel"},
    "Streamlit": {"streamlit"},
    "FastAPI": {"fastapi"},
}


DOMAIN_PHRASES = {
    "Web Development": {
        "web development", "web developer", "web application",
        "web applications", "full stack", "full-stack",
        "frontend development", "front-end development",
        "backend development", "back-end development",
    },
    "Backend Development": {
        "backend development", "back-end development",
        "backend engineering", "server-side development",
        "backend services", "server side development",
    },
    "Frontend Development": {
        "frontend development", "front-end development",
        "user interface development", "ui development",
    },
    "Database Management": {
        "database management", "database development",
        "database design", "relational database",
    },
    "Cloud Computing": {
        "cloud computing", "cloud platform", "cloud platforms",
    },
    "Software Development": {
        "software development", "software engineering",
    },
    "API Development": {
        "api development", "rest api", "restful api", "api integration",
    },
    "Data Engineering": {
        "data engineering", "data pipeline", "data pipelines",
    },
}


PREFERRED_CUES = (
    "preferred",
    "nice to have",
    "nice-to-have",
    "desired",
    "bonus",
    "plus",
    "preferred qualification",
    "preferred qualifications",
)

REQUIRED_CUES = (
    "required",
    "requirements",
    "must have",
    "must-have",
    "mandatory",
    "minimum qualification",
    "minimum qualifications",
    "qualifications",
    "you must",
)


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []

    for value in values:
        text = str(value).strip()
        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _normalize_aliases(text: str) -> str:
    value = str(text or "").lower()

    replacements = {
        "c++": "cplusplus",
        "c#": "csharp",
        ".net": "dotnet",
    }

    for source, target in replacements.items():
        value = value.replace(source, target)

    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _restore_display_term(term: str) -> str:
    replacements = {
        "cplusplus": "C++",
        "csharp": "C#",
        "dotnet": ".NET",
    }

    lower = term.lower().strip()

    if lower in replacements:
        return replacements[lower]

    return term.strip()


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\r\n?", "\n", str(text or ""))
    text = re.sub(r"[ \t]+", " ", text)

    # Preserve line boundaries because many JDs are bullet based.
    raw_parts = re.split(
        r"\n+|(?<=[.!?])\s+|[•●▪◦]+",
        text,
    )

    return [
        re.sub(r"\s+", " ", part).strip()
        for part in raw_parts
        if part and part.strip()
    ]


def _clean_keyword_document(text: str) -> str:
    value = _normalize_aliases(text)

    # Keep technical punctuation, then collapse noisy punctuation.
    value = re.sub(r"[^a-zA-Z0-9+#./\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def _is_generic_phrase(term: str) -> bool:
    words = term.lower().split()

    if not words:
        return True

    if all(word in ALL_STOPWORDS for word in words):
        return True

    generic_fragments = {
        "job description",
        "job role",
        "the candidate",
        "the role",
        "work experience",
        "years experience",
        "relevant experience",
        "required skills",
        "preferred skills",
        "technical skills",
    }

    if term.lower() in generic_fragments:
        return True

    # Reject phrases composed almost entirely of generic words.
    useful = [
        word
        for word in words
        if word not in ALL_STOPWORDS
    ]

    return len(useful) == 0


def _term_has_signal(term: str) -> bool:
    lower = term.lower()

    if _is_generic_phrase(term):
        return False

    if len(term) < 2:
        return False

    if term.isdigit():
        return False

    # Strong technical signal.
    if any(
        symbol in lower
        for symbol in ("+", "#", "/", ".", "-")
    ):
        return True

    # Known skill or domain phrase.
    for aliases in SKILL_ALIASES.values():
        if lower in aliases:
            return True

    for aliases in DOMAIN_PHRASES.values():
        if lower in aliases:
            return True

    # Single natural-language nouns can be useful, but reject common
    # recruiting boilerplate.
    if len(term.split()) == 1 and lower in ALL_STOPWORDS:
        return False

    return True

def _canonicalize_keyword_term(term: str) -> str:
    """
    Normalize extracted aliases to one recruiter-friendly display term.
    """
    value = str(term or "").strip()
    lower = value.lower()

    for canonical, aliases in SKILL_ALIASES.items():
        if lower in {
            _normalize_aliases(alias).lower()
            for alias in aliases
        }:
            return canonical

    for canonical, aliases in DOMAIN_PHRASES.items():
        if lower in {
            _normalize_aliases(alias).lower()
            for alias in aliases
        }:
            return canonical

    return _restore_display_term(value)


def _is_noise_compound_phrase(term: str) -> bool:
    """
    Reject accidental phrases created by n-gram extraction, for example:
        aws docker
        build maintain
        react rest apis

    Known canonical skill phrases and domain phrases remain valid.
    """
    words = [
        word.lower()
        for word in str(term).split()
        if word.strip()
    ]

    if len(words) < 2:
        return False

    normalized_term = _normalize_aliases(term)

    known_phrases = set()

    for aliases in SKILL_ALIASES.values():
        known_phrases.update(
            _normalize_aliases(alias)
            for alias in aliases
        )

    for aliases in DOMAIN_PHRASES.values():
        known_phrases.update(
            _normalize_aliases(alias)
            for alias in aliases
        )

    if normalized_term in known_phrases:
        return False

    action_words = {
        "build", "building", "develop", "developing",
        "maintain", "maintaining", "implement", "implementing",
        "create", "creating", "design", "designing", "manage",
        "managing", "use", "using", "analyze", "analyzing",
        "support", "supporting", "provide", "providing",
        "work", "working", "collaborate", "collaborating",
    }

    if any(word in action_words for word in words):
        return True

    # Reject phrases containing two or more independent known skills.
    known_skill_hits = 0

    for word in words:
        for canonical, aliases in SKILL_ALIASES.items():
            alias_words = {
                _normalize_aliases(alias).lower()
                for alias in aliases
            }

            if (
                word in alias_words
                or word == canonical.lower()
            ):
                known_skill_hits += 1
                break

    if known_skill_hits >= 2:
        return True

    # Avoid generic phrases that just combine a technical token with
    # recruitment boilerplate.
    if any(
        word in ALL_STOPWORDS
        for word in words
    ):
        return True

    return False


def _sentence_tfidf_keywords(
    sentences: Sequence[str],
    n: int,
) -> List[str]:
    if not sentences:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return []

    cleaned_sentences = [
        _clean_keyword_document(sentence)
        for sentence in sentences
        if sentence.strip()
    ]

    cleaned_sentences = [
        sentence
        for sentence in cleaned_sentences
        if sentence
    ]

    if not cleaned_sentences:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(ALL_STOPWORDS),
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#./-]*\b",
            sublinear_tf=True,
            min_df=1,
            max_features=5000,
        )

        matrix = vectorizer.fit_transform(
            cleaned_sentences
        )
    except ValueError:
        return []

    terms = vectorizer.get_feature_names_out()
    scores = matrix.toarray()

    # Average + max rewards terms that carry signal in relevant sentences.
    mean_scores = scores.mean(axis=0)
    max_scores = scores.max(axis=0)

    frequency = Counter()

    for sentence in cleaned_sentences:
        try:
            # A lightweight phrase counter.
            tokens = re.findall(
                r"(?u)\b[a-zA-Z][a-zA-Z0-9+#./-]*\b",
                sentence.lower(),
            )
        except Exception:
            tokens = sentence.lower().split()

        for token in tokens:
            frequency[token] += 1

    ranked: List[Tuple[str, float]] = []

    for index, term in enumerate(terms):
        if not _term_has_signal(term):
            continue

        if _is_noise_compound_phrase(term):
            continue

        lower = term.lower()
        freq = frequency.get(lower, 1)

        # Extra weight for known skills/domain phrases.
        bonus = 1.0

        if any(lower in aliases for aliases in SKILL_ALIASES.values()):
            bonus += 0.45

        if any(lower in aliases for aliases in DOMAIN_PHRASES.values()):
            bonus += 0.30

        if len(term.split()) >= 2:
            bonus += 0.15

        score = (
            0.50 * float(mean_scores[index])
            + 0.30 * float(max_scores[index])
            + 0.08 * min(freq, 5)
        ) * bonus

        ranked.append((term, score))

    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    output = []

    for term, _score in ranked:
        display = _restore_display_term(term)

        # Suppress one-word boilerplate that survived vectorization.
        if len(display.split()) == 1 and display.lower() in ALL_STOPWORDS:
            continue

        output.append(display)

        if len(output) >= max(n * 3, n):
            break

    return _unique(output)[:n]


def extract_detected_skills(text: str) -> List[str]:
    """
    Detect canonical technical skills from the JD.
    """
    normalized = _normalize_aliases(text)
    found = []

    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            alias_normalized = _normalize_aliases(alias)

            if re.search(
                rf"(?<![a-z0-9]){re.escape(alias_normalized)}(?![a-z0-9])",
                normalized,
                flags=re.IGNORECASE,
            ):
                found.append(canonical)
                break

    # Clean duplicates such as NLP + Natural Language Processing.
    # Preserve both only when the JD explicitly contains both forms.
    return _unique(found)


def _classify_skill(
    canonical_skill: str,
    sentences: Sequence[str],
) -> str:
    preferred_hits = 0
    required_hits = 0
    generic_hits = 0

    aliases = SKILL_ALIASES.get(
        canonical_skill,
        {canonical_skill.lower()},
    )

    normalized_aliases = {
        _normalize_aliases(alias)
        for alias in aliases
    }

    for sentence in sentences:
        normalized_sentence = _normalize_aliases(sentence)

        if not any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized_sentence,
                flags=re.IGNORECASE,
            )
            for alias in normalized_aliases
        ):
            continue

        if any(
            cue in normalized_sentence
            for cue in PREFERRED_CUES
        ):
            preferred_hits += 1

        elif any(
            cue in normalized_sentence
            for cue in REQUIRED_CUES
        ):
            required_hits += 1

        else:
            generic_hits += 1

    if preferred_hits > 0 and required_hits == 0:
        return "preferred"

    if required_hits > 0:
        return "required"

    # In a normal JD, a concrete technical skill mentioned without a
    # preferred cue is treated as required for deterministic matching.
    if generic_hits > 0:
        return "required"

    return "required"


def extract_required_and_preferred_skills(
    text: str,
) -> Dict[str, List[str]]:
    sentences = _split_sentences(text)
    skills = extract_detected_skills(text)

    required = []
    preferred = []

    for skill in skills:
        category = _classify_skill(
            skill,
            sentences,
        )

        if category == "preferred":
            preferred.append(skill)
        else:
            required.append(skill)

    return {
        "required_skills": _unique(required),
        "preferred_skills": _unique(preferred),
    }


def extract_experience_requirement(
    text: str,
) -> Dict[str, Any]:
    value = str(text or "")

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
        r"(?:\s+of)?\s+(?:relevant\s+|professional\s+|work\s+)?experience",
        r"(?:minimum|min\.?)\s+"
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
        r"\s+(?:of\s+)?(?:experience|professional experience)",
    ]

    matches = []

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            value,
            flags=re.IGNORECASE,
        ):
            raw = re.sub(
                r"\s+",
                " ",
                match.group(0).strip(),
            )
            try:
                years = float(match.group(1))
            except Exception:
                years = None

            matches.append(
                {
                    "raw": raw,
                    "years": years,
                }
            )

    deduped = []
    seen = set()

    for item in matches:
        key = item["raw"].lower()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    required_years = 0.0

    numeric_values = [
        item["years"]
        for item in deduped
        if item["years"] is not None
    ]

    if numeric_values:
        required_years = max(
            value
            for value in numeric_values
            if 0 <= value <= 50
        )

    return {
        "required_years": round(required_years, 2),
        "raw_matches": [
            item["raw"]
            for item in deduped
        ],
    }


def extract_job_title(text: str) -> str:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in str(text or "").splitlines()
        if line.strip()
    ]

    patterns = [
        r"^(?:job\s*title|position|role|title)\s*[:\-]\s*(.+)$",
        r"^(?:position\s*title)\s*[:\-]\s*(.+)$",
    ]

    for line in lines[:30]:
        for pattern in patterns:
            match = re.match(
                pattern,
                line,
                flags=re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip()
                if 2 <= len(value) <= 100:
                    return value

    # A conservative fallback: headings containing developer/engineer/
    # analyst/scientist/designer/manager.
    role_words = (
        "developer", "engineer", "analyst", "scientist",
        "designer", "architect", "manager", "consultant",
    )

    for line in lines[:25]:
        lower = line.lower()
        if any(word in lower for word in role_words):
            if len(line.split()) <= 8 and len(line) <= 100:
                return line

    return ""


def extract_top_jd_keywords(
    text: str,
    n: int = 20,
) -> List[str]:
    """
    Clean JD keyword extraction for recruiter display.

    This is intentionally separate from ATS keyword scoring so the UI
    can show human-readable keywords without changing deterministic
    resume/JD scoring.
    """
    if not text:
        return []

    sentences = _split_sentences(text)
    candidate_terms = _sentence_tfidf_keywords(
        sentences,
        n=max(10, int(n) * 2),
    )

    # Always surface detected technical skills early.
    detected_skills = extract_detected_skills(text)

    # Add normalized domain phrases when explicitly present.
    normalized_text = _normalize_aliases(text)
    domain_terms = []

    for canonical, aliases in DOMAIN_PHRASES.items():
        if any(
            alias in normalized_text
            for alias in (
                _normalize_aliases(item)
                for item in aliases
            )
        ):
            domain_terms.append(canonical)

    merged = _unique(
        list(detected_skills)
        + list(domain_terms)
        + list(candidate_terms)
    )

    # Final phrase-level quality pass.
    cleaned = []

    for item in merged:
        canonical = _canonicalize_keyword_term(item)

        if _is_generic_phrase(canonical):
            continue

        if _is_noise_compound_phrase(canonical):
            continue

        words = canonical.split()

        if len(words) == 1:
            lower = words[0].lower()

            # Single-word output is intentionally strict.
            known = any(
                lower in {
                    _normalize_aliases(alias)
                    for alias in aliases
                }
                or lower == canonical_name.lower()
                for canonical_name, aliases in SKILL_ALIASES.items()
            )

            technical = any(
                marker in lower
                for marker in (
                    "+", "#", ".", "/", "-",
                )
            )

            if (
                lower in ALL_STOPWORDS
                or (not known and not technical)
            ):
                continue

        else:
            # For recruiter-facing JD Keywords, only keep multi-word
            # phrases that are recognized technical/domain concepts.
            # This prevents accidental n-grams such as "react rest".
            normalized_canonical = _normalize_aliases(canonical)

            known_phrase = any(
                normalized_canonical
                in {
                    _normalize_aliases(alias)
                    for alias in aliases
                }
                for aliases in (
                    list(SKILL_ALIASES.values())
                    + list(DOMAIN_PHRASES.values())
                )
            )

            if not known_phrase:
                continue

        cleaned.append(canonical)

    return _unique(cleaned)[:max(1, int(n))]


def build_keyword_groups(
    text: str,
    keywords: Sequence[str],
) -> Dict[str, List[str]]:
    technical_set = {
        skill.lower()
        for skill in extract_detected_skills(text)
    }

    domain_set = set()

    normalized_text = _normalize_aliases(text)

    for canonical, aliases in DOMAIN_PHRASES.items():
        if any(
            _normalize_aliases(alias) in normalized_text
            for alias in aliases
        ):
            domain_set.add(canonical.lower())

    technical = []
    domain = []
    concepts = []

    for keyword in keywords:
        lower = keyword.lower()

        if lower in technical_set:
            technical.append(keyword)
        elif lower in domain_set:
            domain.append(keyword)
        elif len(keyword.split()) >= 2:
            concepts.append(keyword)
        else:
            concepts.append(keyword)

    return {
        "technical": _unique(technical),
        "domain": _unique(domain),
        "concepts": _unique(concepts),
    }


# ------------------------------------------------------------------
# Local retrieval for JD evidence
# ------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9+#.\-]{1,}",
            str(text or "").lower(),
        )
    )


def _keyword_overlap(
    query: str,
    text: str,
) -> float:
    q = _token_set(query)
    d = _token_set(text)

    if not q or not d:
        return 0.0

    return len(q & d) / len(q)


def retrieve_jd_evidence(
    query: str,
    jd_text: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant JD chunks using local TF-IDF + overlap.
    This is the retrieval half of the optional RAG pipeline.
    """
    sentences = _split_sentences(jd_text)

    if not sentences:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return []

    documents = [
        _clean_keyword_document(sentence)
        for sentence in sentences
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(ALL_STOPWORDS),
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000,
        )

        matrix = vectorizer.fit_transform(
            documents + [
                _clean_keyword_document(query)
            ]
        )

        similarities = cosine_similarity(
            matrix[-1],
            matrix[:-1],
        )[0]
    except ValueError:
        return []

    scored = []

    for index, sentence in enumerate(sentences):
        overlap = _keyword_overlap(
            query,
            sentence,
        )

        score = (
            0.75 * float(similarities[index])
            + 0.25 * float(overlap)
        )

        scored.append(
            {
                "id": index + 1,
                "text": sentence,
                "tfidf_score": round(
                    float(similarities[index]),
                    4,
                ),
                "keyword_overlap": round(
                    float(overlap),
                    4,
                ),
                "retrieval_score": round(
                    float(score),
                    4,
                ),
            }
        )

    scored.sort(
        key=lambda item: item["retrieval_score"],
        reverse=True,
    )

    return scored[:max(1, int(top_k))]


# ------------------------------------------------------------------
# Optional Gemini enrichment
# ------------------------------------------------------------------

GEMINI_JD_MODEL = os.getenv(
    "GEMINI_JD_MODEL",
    os.getenv(
        "GEMINI_SCREENING_MODEL",
        "gemini-3.5-flash-lite",
    ),
)

GEMINI_JD_FALLBACKS = [
    GEMINI_JD_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
]

JD_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "technical_keywords": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "domain_keywords": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "concept_keywords": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "technical_keywords",
        "domain_keywords",
        "concept_keywords",
    ],
}


def _retryable_gemini_error(
    error: Exception,
) -> bool:
    text = str(error).lower()

    return any(
        marker in text
        for marker in (
            "503",
            "unavailable",
            "high demand",
            "temporarily",
            "resource_exhausted",
        )
    )


def enrich_keywords_with_rag(
    jd_text: str,
    *,
    top_k: int = 6,
    max_keywords: int = 20,
    api_key: str | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Retrieve JD evidence locally, then optionally ask Gemini to normalize
    and enrich keywords using only those retrieved JD snippets.
    """
    api_key = (
        api_key
        if api_key is not None
        else os.getenv("GEMINI_API_KEY", "")
    ).strip()

    model = (
        model
        or GEMINI_JD_MODEL
    ).strip()

    query = (
        "technical skills technologies frameworks "
        "tools platforms programming languages domain concepts "
        "required preferred qualifications"
    )

    retrieved = retrieve_jd_evidence(
        query,
        jd_text,
        top_k=top_k,
    )

    if not retrieved:
        return {
            "keywords": extract_top_jd_keywords(
                jd_text,
                n=max_keywords,
            ),
            "groups": {},
            "retrieved": [],
            "used_gemini": False,
            "model": None,
            "warning": "No JD evidence could be retrieved.",
        }

    if not api_key:
        local = extract_top_jd_keywords(
            jd_text,
            n=max_keywords,
        )

        return {
            "keywords": local,
            "groups": build_keyword_groups(
                jd_text,
                local,
            ),
            "retrieved": retrieved,
            "used_gemini": False,
            "model": None,
            "warning": "GEMINI_API_KEY is not configured; local retrieval was used.",
        }

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        local = extract_top_jd_keywords(
            jd_text,
            n=max_keywords,
        )

        return {
            "keywords": local,
            "groups": build_keyword_groups(
                jd_text,
                local,
            ),
            "retrieved": retrieved,
            "used_gemini": False,
            "model": None,
            "warning": "google-genai is not installed; local retrieval was used.",
        }

    evidence = "\n\n".join(
        f"[JD EVIDENCE {item['id']}]\n{item['text']}"
        for item in retrieved
    )

    prompt = f"""
You are extracting recruitment-relevant keywords from a Job Description.

Use ONLY the supplied JD evidence.

RULES:
1. Remove stopwords and generic recruitment boilerplate.
2. Prefer concrete skills, technologies, frameworks, tools, platforms,
   methodologies, domain terms, and meaningful multi-word concepts.
3. Do not return generic words such as candidate, role, experience,
   responsibility, skills, required, preferred, company, team, etc.
4. Normalize obvious aliases to common names.
5. Do not invent skills that are not present in the evidence.
6. Return concise keyword phrases, not sentences.
7. Return at most {max_keywords} total items.

JD EVIDENCE:
{evidence}
""".strip()

    response = None
    used_model = model
    last_error = None

    try:
        client = genai.Client(api_key=api_key)

        model_candidates = _unique(
            [model] + GEMINI_JD_FALLBACKS
        )

        for candidate_model in model_candidates:
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=JD_RESPONSE_SCHEMA,
                        temperature=0.1,
                        max_output_tokens=700,
                    ),
                )
                used_model = candidate_model
                break
            except Exception as error:
                last_error = error

                if not _retryable_gemini_error(error):
                    raise

        if response is None:
            raise RuntimeError(
                f"All configured Gemini models failed. Last error: {last_error}"
            )

        response_text = getattr(
            response,
            "text",
            "",
        ) or ""

        parsed = json.loads(
            response_text
        )

        groups = {
            "technical": _unique(
                parsed.get("technical_keywords", [])
            ),
            "domain": _unique(
                parsed.get("domain_keywords", [])
            ),
            "concepts": _unique(
                parsed.get("concept_keywords", [])
            ),
        }

        merged = _unique(
            groups["technical"]
            + groups["domain"]
            + groups["concepts"]
        )

        local = extract_top_jd_keywords(
            jd_text,
            n=max_keywords,
        )

        # Keep deterministic technical hits even if Gemini omits one.
        final_keywords = _unique(
            groups["technical"]
            + extract_detected_skills(jd_text)
            + groups["domain"]
            + groups["concepts"]
            + local
        )[:max_keywords]

        return {
            "keywords": final_keywords,
            "groups": groups,
            "retrieved": retrieved,
            "used_gemini": True,
            "model": used_model,
            "warning": "",
        }

    except Exception as error:
        local = extract_top_jd_keywords(
            jd_text,
            n=max_keywords,
        )

        return {
            "keywords": local,
            "groups": build_keyword_groups(
                jd_text,
                local,
            ),
            "retrieved": retrieved,
            "used_gemini": False,
            "model": None,
            "warning": (
                f"Gemini enrichment unavailable; local extraction was used. "
                f"Reason: {error}"
            ),
        }


# ------------------------------------------------------------------
# Main JD analysis API
# ------------------------------------------------------------------

def analyze_job_description(
    text: str,
) -> Dict[str, Any]:
    """
    Backwards-compatible JD intelligence API.

    Existing matching code can continue using:
        required_skills
        preferred_skills
        experience
        job_title
    """
    if not text:
        return {
            "job_title": "",
            "title": "",
            "role": "",
            "required_skills": [],
            "preferred_skills": [],
            "experience": {
                "required_years": 0.0,
                "raw_matches": [],
            },
            "keywords": [],
            "keyword_groups": {
                "technical": [],
                "domain": [],
                "concepts": [],
            },
            "detected_skills": [],
            "source": "Hybrid JD Intelligence V2",
        }

    skills = extract_required_and_preferred_skills(
        text
    )

    keywords = extract_top_jd_keywords(
        text,
        n=20,
    )

    title = extract_job_title(text)

    experience = extract_experience_requirement(
        text
    )

    return {
        "job_title": title,
        "title": title,
        "role": title,
        "required_skills": skills["required_skills"],
        "preferred_skills": skills["preferred_skills"],
        "experience": experience,
        "keywords": keywords,
        "keyword_groups": build_keyword_groups(
            text,
            keywords,
        ),
        "detected_skills": extract_detected_skills(
            text
        ),
        "source": "Hybrid JD Intelligence V2",
    }


# Backwards-compatible alias for callers that previously imported this
# module's keyword extractor.
extract_top_keywords = extract_top_jd_keywords


if __name__ == "__main__":
    SAMPLE_JD = """
    Web Developer

    Required:
    2+ years of experience with JavaScript, Python, SQL, HTML, CSS,
    React and REST APIs. Experience with Oracle databases is required.

    Preferred:
    AWS, Docker and GitHub experience is a plus.

    Responsibilities:
    Build and maintain web applications and backend services.
    """

    result = analyze_job_description(
        SAMPLE_JD
    )

    print(json.dumps(result, indent=2))

    print("\nCLEAN KEYWORDS:")
    print(", ".join(result["keywords"]))

    print("\nRAG-READY RETRIEVAL TEST:")
    retrieved = retrieve_jd_evidence(
        "skills technologies frameworks databases web development",
        SAMPLE_JD,
        top_k=4,
    )

    for item in retrieved:
        print(
            f"- {item['retrieval_score']:.3f}: "
            f"{item['text']}"
        )
