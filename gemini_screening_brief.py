"""
Free Gemini LLM Screening Brief
--------------------------------

Uses Google's Gemini Developer API through the official google-genai SDK.

Default model:
    gemini-3.8-flash

The free tier is intended for experimentation/small projects. Quotas are
project/model dependent and can change, so the app surfaces errors rather
than pretending unlimited usage.

Design:
- Deterministic ATS/requirement scores remain unchanged.
- Only already-computed, job-related evidence is sent to Gemini.
- The model is instructed not to invent facts.
- JSON output is constrained with a response schema.
- No OpenAI dependency.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_MODEL = os.getenv(
    "GEMINI_SCREENING_MODEL",
    "gemini-3.5-flash-lite",
)

MAX_EVIDENCE_CHARS = 18000

FREE_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
]


def _is_retryable_capacity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "503" in text
        or "unavailable" in text
        or "high demand" in text
        or "temporarily" in text
        or "resource_exhausted" in text
    )


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "executive_summary": {
            "type": "STRING",
        },
        "strengths": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "gaps_and_verification": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "interview_focus": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "recruiter_talking_points": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "suggested_questions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "evidence_notes": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "executive_summary",
        "strengths",
        "gaps_and_verification",
        "interview_focus",
        "recruiter_talking_points",
        "suggested_questions",
        "evidence_notes",
    ],
}


def get_api_key() -> str:
    """
    Read GEMINI_API_KEY from the process environment.
    """
    return os.getenv("GEMINI_API_KEY", "").strip()


def is_configured() -> bool:
    return bool(get_api_key())


def build_evidence_payload(
    *,
    candidate_row: Dict[str, Any],
    analysis_result: Dict[str, Any],
    ats_result: Dict[str, Any],
    intelligence: Dict[str, Any],
    quality: Dict[str, Any],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:

    required = requirements.get(
        "required_skills",
        {},
    ) or {}

    preferred = requirements.get(
        "preferred_skills",
        {},
    ) or {}

    return {
        "candidate": {
            "name": candidate_row.get("Candidate"),
            "resume": candidate_row.get("Resume"),
            "experience_years": intelligence.get(
                "experience_years"
            ),
            "education": intelligence.get("education"),
            "projects": intelligence.get("projects"),
            "certifications": intelligence.get(
                "certifications"
            ),
            "all_skills": intelligence.get(
                "all_skills"
            ),
            "skills_by_category": intelligence.get(
                "skills"
            ),
            "section_completeness": intelligence.get(
                "section_completeness"
            ),
            "contact_completeness": intelligence.get(
                "contact_completeness"
            ),
        },
        "ats": {
            "ats_score": ats_result.get("ats_score"),
            "skill_score": ats_result.get("skill_score"),
            "semantic_score": ats_result.get("semantic_score"),
            "lexical_score": ats_result.get("lexical_score"),
            "keyword_score": ats_result.get("keyword_score"),
            "strengths": ats_result.get("strengths"),
            "suggestions": ats_result.get("suggestions"),
        },
        "matching": {
            "matched_skills": analysis_result.get(
                "matched_skills"
            ),
            "missing_skills": analysis_result.get(
                "missing_skills"
            ),
            "matched_keywords": analysis_result.get(
                "matched_keywords"
            ),
            "missing_keywords": analysis_result.get(
                "missing_keywords"
            ),
            "section_scores": analysis_result.get(
                "section_scores"
            ),
        },
        "requirements": {
            "requirement_match_score": requirements.get(
                "requirement_match_score"
            ),
            "required_skills": {
                "matched": required.get("matched"),
                "missing": required.get("missing"),
                "score": required.get("score"),
            },
            "preferred_skills": {
                "matched": preferred.get("matched"),
                "missing": preferred.get("missing"),
                "score": preferred.get("score"),
            },
            "experience": requirements.get("experience"),
            "education": requirements.get("education"),
        },
        "resume_quality": {
            "quality_score": quality.get("quality_score"),
            "action_verbs": quality.get("action_verbs"),
            "quantification": quality.get("quantification"),
            "repetition": quality.get("repetition"),
            "length": quality.get("length"),
            "recommendations": quality.get("recommendations"),
            "extraction_issues": quality.get(
                "extraction_issues"
            ),
        },
    }


def _build_prompt(
    evidence: Dict[str, Any]
) -> str:

    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    if len(evidence_json) > MAX_EVIDENCE_CHARS:
        evidence_json = (
            evidence_json[:MAX_EVIDENCE_CHARS]
            + "\n...[evidence truncated]"
        )

    return f"""
You are an evidence-grounded recruiting assistant.

Create a concise screening brief from the supplied structured evidence.

STRICT RULES:
1. Use only the supplied evidence.
2. Never invent employers, job titles, dates, degrees,
   certifications, skills, projects, achievements, or interview results.
3. "Not detected" means the information was not detected in the supplied
   evidence; it does not mean the candidate does not possess it.
4. Do not recompute or modify any score.
5. Do not make a hiring decision.
6. Keep observations job-related.
7. Suggested interview questions should validate evidence or resolve
   uncertainty.
8. Keep each list concise, with at most 6 items.
9. Return a professional recruiter-facing brief.

SUPPLIED EVIDENCE:
{evidence_json}
""".strip()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, Iterable):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    value = str(value).strip()
    return [value] if value else []


def _clean_output(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "executive_summary",
        "strengths",
        "gaps_and_verification",
        "interview_focus",
        "recruiter_talking_points",
        "suggested_questions",
        "evidence_notes",
    ]

    output: Dict[str, Any] = {}

    for key in keys:
        if key == "executive_summary":
            output[key] = str(
                data.get(key, "")
                or ""
            ).strip()
            continue

        seen = set()
        values = []

        for item in _as_list(data.get(key, [])):
            normalized = item.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            values.append(item)

        output[key] = values[:6]

    if not output["executive_summary"]:
        output["executive_summary"] = (
            "No executive summary was returned. "
            "Review the structured candidate evidence."
        )

    return output


def generate_gemini_screening_brief(
    *,
    candidate_row: Dict[str, Any],
    analysis_result: Dict[str, Any],
    ats_result: Dict[str, Any],
    intelligence: Dict[str, Any],
    quality: Dict[str, Any],
    requirements: Dict[str, Any],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a grounded screening brief using Gemini Developer API.
    """

    api_key = (
        api_key
        if api_key is not None
        else get_api_key()
    )
    api_key = str(api_key).strip()

    model = str(
        model or DEFAULT_MODEL
    ).strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Google GenAI SDK is not installed. "
            "Run: pip install google-genai"
        ) from exc

    evidence = build_evidence_payload(
        candidate_row=candidate_row,
        analysis_result=analysis_result,
        ats_result=ats_result,
        intelligence=intelligence,
        quality=quality,
        requirements=requirements,
    )

    prompt = _build_prompt(evidence)

    try:
        client = genai.Client(
            api_key=api_key
        )

        # Try the configured model first. If it is temporarily unavailable
        # (for example, a 503 capacity/high-demand response), automatically
        # fall back to another free-tier model.
        models_to_try = [model] + [
            item
            for item in FREE_FALLBACK_MODELS
            if item != model
        ]

        response = None
        last_error = None
        used_model = model

        for candidate_model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                        temperature=0.2,
                        max_output_tokens=1200,
                    ),
                )
                used_model = candidate_model
                break
            except Exception as exc:
                last_error = exc

                if not _is_retryable_capacity_error(exc):
                    raise

        if response is None:
            raise RuntimeError(
                "All configured free Gemini models were temporarily "
                f"unavailable. Last error: {last_error}"
            )

    except Exception as exc:
        raise RuntimeError(
            f"Gemini API request failed: {exc}"
        ) from exc

    response_text = getattr(
        response,
        "text",
        "",
    ) or ""

    if not response_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:
        parsed = json.loads(
            response_text
        )
    except json.JSONDecodeError as exc:
        # Small fallback if the provider wraps JSON in fences.
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            response_text.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(
                "Gemini returned invalid JSON."
            ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Gemini returned an unexpected response format."
        )

    result = _clean_output(parsed)

    result["candidate"] = (
        str(
            candidate_row.get(
                "Candidate",
                "Candidate",
            )
            or "Candidate"
        ).strip()
    )

    result["model"] = used_model
    result["source"] = "Google Gemini Developer API"
    result["fallback_used"] = used_model != model
    result["grounded"] = True

    return result


def gemini_brief_to_text(
    brief: Dict[str, Any]
) -> str:

    lines = [
        "AI-ASSISTED CANDIDATE SCREENING BRIEF",
        "=" * 55,
        f"Candidate: {brief.get('candidate', 'Candidate')}",
        f"Model: {brief.get('model', DEFAULT_MODEL)}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 22,
        str(
            brief.get(
                "executive_summary",
                "",
            )
        ),
    ]

    sections = [
        ("KEY STRENGTHS", "strengths"),
        ("GAPS / VERIFICATION AREAS", "gaps_and_verification"),
        ("INTERVIEW FOCUS", "interview_focus"),
        ("RECRUITER TALKING POINTS", "recruiter_talking_points"),
        ("SUGGESTED INTERVIEW QUESTIONS", "suggested_questions"),
        ("EVIDENCE NOTES", "evidence_notes"),
    ]

    for title, key in sections:
        lines.extend(
            [
                "",
                title,
                "-" * 22,
            ]
        )

        values = _as_list(
            brief.get(key, [])
        )

        if values:
            lines.extend(
                f"- {item}"
                for item in values
            )
        else:
            lines.append("None identified.")

    lines.extend(
        [
            "",
            "Grounding note: This brief summarizes structured evidence "
            "already calculated by the CV Analyzer. Verify important "
            "details against the original resume and interview evidence.",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # Offline smoke test: verifies the module can load without making
    # an API request.
    print(
        "Gemini screening brief engine is ready. "
        "No API call was made."
    )
