"""Business/service layer for the FastAPI backend.

The service reuses the existing CV Analyzer intelligence, ATS, matching,
requirement-matrix and explainability modules instead of creating a second
scoring implementation.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _jsonable(value: Any) -> Any:
    """Convert common Python/NumPy/Pandas values into JSON-safe objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    # Pandas DataFrame / Series without requiring pandas at import time.
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(orient="records"))
        except TypeError:
            try:
                return _jsonable(value.to_dict())
            except Exception:
                pass

    # NumPy scalar support without requiring numpy at import time.
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass

    return str(value)


def _lazy_import_pipeline():
    """Import heavy project modules only when an analysis is requested."""
    from analysis import analyze_resumes_batch
    from ats_engine import analyze_ats
    from jd_intelligence_v6 import analyze_job_description
    from matching_engine import analyze_candidate_requirements
    from resume_intelligence import analyze_resume_intelligence
    from resume_quality import analyze_resume_quality
    from jd_requirement_matrix import (
        build_candidate_requirement_matrix,
        build_requirement_matrix,
    )
    from explainability_engine import build_explainability

    return {
        "analyze_resumes_batch": analyze_resumes_batch,
        "analyze_ats": analyze_ats,
        "analyze_job_description": analyze_job_description,
        "analyze_candidate_requirements": analyze_candidate_requirements,
        "analyze_resume_intelligence": analyze_resume_intelligence,
        "analyze_resume_quality": analyze_resume_quality,
        "build_candidate_requirement_matrix": build_candidate_requirement_matrix,
        "build_requirement_matrix": build_requirement_matrix,
        "build_explainability": build_explainability,
    }


def extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """Extract text from TXT/MD, PDF or DOCX bytes."""
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        return raw_bytes.decode("utf-8", errors="replace").strip()

    if suffix == ".pdf":
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages).strip()

    if suffix == ".docx":
        import docx2txt

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            return (docx2txt.process(tmp_path) or "").strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    raise ValueError(
        f"Unsupported file type '{suffix or 'unknown'}'. "
        "Use PDF, DOCX, TXT or Markdown."
    )


def _candidate_payload(
    *,
    resume_name: str,
    resume_text: str,
    analysis_result: Dict[str, Any],
    pipeline: Dict[str, Any],
    jd_analysis: Dict[str, Any],
    include_raw_text: bool = False,
) -> Dict[str, Any]:
    """Build the complete candidate result using the existing V9 signals."""
    analyze_ats = pipeline["analyze_ats"]
    analyze_resume_intelligence = pipeline["analyze_resume_intelligence"]
    analyze_resume_quality = pipeline["analyze_resume_quality"]
    analyze_candidate_requirements = pipeline["analyze_candidate_requirements"]
    build_candidate_requirement_matrix = pipeline["build_candidate_requirement_matrix"]
    build_explainability = pipeline["build_explainability"]

    ats_result = analyze_ats(
        skill_score=analysis_result.get("skill_score", 0),
        semantic_score=analysis_result.get("semantic_score", 0),
        lexical_score=analysis_result.get("lexical_score", 0),
        matched_keywords=analysis_result.get("matched_keywords", []),
        missing_keywords=analysis_result.get("missing_keywords", []),
        matched_skills=analysis_result.get("matched_skills", []),
        missing_skills=analysis_result.get("missing_skills", []),
    )

    intelligence = analyze_resume_intelligence(
        resume_text,
        source_filename=resume_name,
    )
    quality = analyze_resume_quality(
        resume_text,
        sections=intelligence.get("sections", {}),
    )
    requirements = analyze_candidate_requirements(
        candidate_intelligence=intelligence,
        jd_analysis=jd_analysis,
    )

    candidate_matrix = build_candidate_requirement_matrix(
        jd_analysis=jd_analysis,
        candidate_requirements=requirements,
        candidate_intelligence=intelligence,
    )

    explanation = build_explainability(
        resume_text=resume_text,
        jd_text=jd_analysis.get("raw_text", ""),
        analysis_result=analysis_result,
        ats_result=ats_result,
        intelligence=intelligence,
        quality=quality,
        requirements=requirements,
        candidate_matrix=candidate_matrix,
    )

    result: Dict[str, Any] = {
        "resume_name": resume_name,
        "candidate_name": str(intelligence.get("name", "")).strip(),
        "analysis": analysis_result,
        "ats": ats_result,
        "intelligence": intelligence,
        "quality": quality,
        "requirements": requirements,
        "requirement_matrix": candidate_matrix,
        "explainability": explanation,
        "contact": {
            "email": intelligence.get("email", ""),
            "phone": intelligence.get("phone", ""),
            "linkedin": intelligence.get("linkedin", ""),
            "github": intelligence.get("github", ""),
        },
    }

    if include_raw_text:
        result["resume_text"] = resume_text

    return _jsonable(result)


def analyze_candidates(
    job_description: str,
    resumes: Sequence[Tuple[str, str]],
    *,
    top_n_keywords: int = 20,
    include_raw_text: bool = False,
) -> Dict[str, Any]:
    """Run the existing CV Analyzer pipeline for one or more candidates."""
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")
    if not resumes:
        raise ValueError("At least one resume is required.")

    clean_resumes = []
    for name, text in resumes:
        name = str(name).strip() or "resume.txt"
        text = str(text or "").strip()
        if text:
            clean_resumes.append((name, text))

    if not clean_resumes:
        raise ValueError("No readable resume text was provided.")

    pipeline = _lazy_import_pipeline()
    analyze_resumes_batch = pipeline["analyze_resumes_batch"]
    analyze_job_description = pipeline["analyze_job_description"]
    build_requirement_matrix = pipeline["build_requirement_matrix"]

    jd_analysis = _jsonable(analyze_job_description(job_description))
    # Preserve the original JD text for the V9 evidence engine without
    # changing the deterministic JD schema.
    jd_analysis["raw_text"] = job_description
    jd_matrix = build_requirement_matrix(jd_analysis)

    batch_results = analyze_resumes_batch(
        job_description,
        clean_resumes,
        top_n=top_n_keywords,
    )

    lookup = {
        str(result.get("resume_name", "")): result
        for result in (batch_results or [])
    }

    candidates: List[Dict[str, Any]] = []
    for resume_name, resume_text in clean_resumes:
        analysis_result = lookup.get(resume_name)
        if not analysis_result:
            continue
        try:
            candidates.append(
                _candidate_payload(
                    resume_name=resume_name,
                    resume_text=resume_text,
                    analysis_result=analysis_result,
                    pipeline=pipeline,
                    jd_analysis=jd_analysis,
                    include_raw_text=include_raw_text,
                )
            )
        except Exception as exc:
            candidates.append(
                {
                    "resume_name": resume_name,
                    "error": str(exc),
                }
            )

    if not candidates:
        raise RuntimeError("The analysis engine returned no candidate results.")

    return _jsonable(
        {
            "job_description": {
                "analysis": jd_analysis,
                "requirement_matrix": jd_matrix,
            },
            "candidates": candidates,
            "meta": {
                "candidate_count": len(candidates),
                "requested_candidate_count": len(clean_resumes),
                "top_n_keywords": top_n_keywords,
                "pipeline": "CV Analyzer V10 FastAPI + existing V9 engines",
            },
        }
    )
