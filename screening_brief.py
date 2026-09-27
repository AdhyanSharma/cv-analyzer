"""
AI-Assisted Candidate Screening Brief

Turns the existing ATS, requirement, skills, semantic and quality
signals into a concise recruiter-facing brief.

No new third-party dependency.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(x).strip() for x in value if str(x).strip()]
    value = str(value).strip()
    return [value] if value else []


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        item = str(value).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _join(values: Iterable[str], limit: int = 8) -> str:
    vals = _unique(values)
    if not vals:
        return "None detected"
    if len(vals) <= limit:
        return ", ".join(vals)
    return ", ".join(vals[:limit]) + f" (+{len(vals)-limit} more)"


def build_screening_brief(
    *,
    candidate_row: Dict[str, Any],
    analysis_result: Dict[str, Any],
    ats_result: Dict[str, Any],
    intelligence: Dict[str, Any],
    quality: Dict[str, Any],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    candidate = _text(candidate_row.get("Candidate")) or "Candidate"

    ats = _float(ats_result.get("ats_score", candidate_row.get("ATS Score", 0)))
    semantic = _float(ats_result.get("semantic_score", candidate_row.get("Semantic Score", 0)))
    skill = _float(ats_result.get("skill_score", candidate_row.get("Skill Score", 0)))
    lexical = _float(ats_result.get("lexical_score", candidate_row.get("Lexical Score", 0)))
    keyword = _float(ats_result.get("keyword_score", candidate_row.get("Keyword Score", 0)))
    quality_score = _float(quality.get("quality_score", candidate_row.get("Resume Quality", 0)))

    req_value = requirements.get("requirement_match_score")
    requirement = _float(req_value) if req_value is not None else None

    experience = _float(
        intelligence.get("experience_years", candidate_row.get("Experience Years", 0))
    )

    all_skills = _list(intelligence.get("all_skills", []))
    matched_skills = _list(analysis_result.get("matched_skills", []))
    missing_skills = _list(analysis_result.get("missing_skills", []))
    matched_keywords = _list(analysis_result.get("matched_keywords", []))

    required = requirements.get("required_skills", {}) or {}
    preferred = requirements.get("preferred_skills", {}) or {}

    required_matched = _list(required.get("matched", []))
    required_missing = _list(required.get("missing", []))
    preferred_matched = _list(preferred.get("matched", []))
    preferred_missing = _list(preferred.get("missing", []))

    exp_info = requirements.get("experience", {}) or {}
    required_experience = _float(exp_info.get("required_years", 0))
    experience_status = _text(exp_info.get("status")) or "Not specified"

    education_status = _text(
        (requirements.get("education", {}) or {}).get("status")
    ) or "Not specified"

    sections = intelligence.get("sections", {}) or {}
    section_completeness = _float(intelligence.get("section_completeness", 0))
    contact_completeness = _float(intelligence.get("contact_completeness", 0))

    quant_score = _float(
        (quality.get("quantification", {}) or {}).get("score")
    )
    repetition_score = _float(
        (quality.get("repetition", {}) or {}).get("score")
    )

    strengths = []
    gaps = []
    focus = []
    talking_points = []
    questions = []
    evidence = []

    # Strengths
    if skill >= 80:
        strengths.append(f"Strong extracted skill coverage ({skill:.1f}%).")
    elif skill >= 60:
        strengths.append(f"Moderate extracted skill coverage ({skill:.1f}%).")

    if semantic >= 80:
        strengths.append(f"High semantic alignment with the Job Description ({semantic:.1f}%).")
    elif semantic >= 60:
        strengths.append(f"Good semantic alignment with the Job Description ({semantic:.1f}%).")

    if required_matched:
        strengths.append("Matched required skills: " + _join(required_matched))

    if preferred_matched:
        strengths.append("Matched preferred skills: " + _join(preferred_matched))

    if experience > 0:
        strengths.append(f"Approximately {experience:.1f} years of experience are explicitly detected.")

    if experience_status and experience_status != "Not specified":
        strengths.append(f"Experience requirement signal: {experience_status}.")

    if quality_score >= 85:
        strengths.append(f"Resume quality indicators are strong ({quality_score:.1f}%).")
    elif quality_score >= 70:
        strengths.append(f"Resume quality indicators are generally solid ({quality_score:.1f}%).")

    if quant_score >= 70:
        strengths.append("Useful quantified achievement evidence was detected.")

    if sections.get("projects"):
        strengths.append("Projects section detected for technical discussion.")

    if sections.get("certifications"):
        strengths.append("Certifications section detected.")

    strengths = _unique(strengths)

    # Gaps / verification
    if required_missing:
        gaps.append("Required skills not detected: " + _join(required_missing))

    if preferred_missing:
        gaps.append("Preferred skills not detected: " + _join(preferred_missing))

    extra_missing = [
        s for s in missing_skills
        if s.lower() not in {x.lower() for x in required_missing}
    ]
    if extra_missing:
        gaps.append("Additional JD skill gaps: " + _join(extra_missing))

    if required_experience > 0 and experience < required_experience:
        gaps.append(
            f"Explicit experience is below the extracted requirement "
            f"({experience:.1f} vs {required_experience:.1f} years)."
        )

    if contact_completeness < 100:
        gaps.append(f"Contact information completeness is {contact_completeness:.1f}%.")

    if section_completeness < 70:
        gaps.append(
            f"Section completeness is {section_completeness:.1f}%; "
            "some standard resume sections may be missing."
        )

    if quant_score < 50:
        gaps.append("Limited quantified achievement evidence was detected.")

    if repetition_score < 60:
        gaps.append("Some repetition or low-variety wording was detected.")

    if not sections.get("projects"):
        gaps.append(
            "No dedicated Projects section was detected; clarify project experience during screening."
        )

    if education_status not in {"Not specified", ""}:
        gaps.append("Education requirement signal: " + education_status)

    gaps = _unique(gaps)

    # Interview focus
    if required_missing:
        focus.append("Validate real-world exposure to the missing required skills.")

    if preferred_missing:
        focus.append("Ask whether preferred skills were developed through projects, coursework, or work.")

    if sections.get("projects"):
        focus.append(
            "Deep-dive into one project: architecture, contribution, trade-offs, testing, and measurable outcome."
        )
    else:
        focus.append(
            "Ask for the most technically challenging problem the candidate has solved."
        )

    if semantic < 60:
        focus.append("Verify role relevance because semantic alignment is relatively low.")

    if lexical < 50:
        focus.append("Clarify responsibilities that may use terminology different from the Job Description.")

    if quality_score < 70:
        focus.append("Use interview evidence to validate experience that is weakly represented in the resume.")

    focus = _unique(focus)

    if not focus:
        focus = [
            "Validate the strongest role-relevant skills with concrete examples.",
            "Walk through one recent technical project or work item.",
            "Confirm ownership, decision-making, and measurable impact.",
        ]

    # Recruiter talking points
    talking_points.append("Start with the candidate's most relevant experience and ask for one concrete example.")

    if matched_skills:
        talking_points.append(
            "Discuss matched skills in depth: " + _join(matched_skills) + "."
        )

    if matched_keywords:
        talking_points.append(
            "Connect experience to JD terms such as " + _join(matched_keywords, 6) + "."
        )

    if missing_skills:
        talking_points.append(
            "Clarify whether missing skills are absent or simply not stated explicitly in the resume."
        )

    talking_points.append(
        "Ask for measurable outcomes rather than only listing tools or technologies."
    )
    talking_points = _unique(talking_points)

    # Interview questions
    for skill_name in required_missing[:3]:
        questions.append(
            f"Can you describe a project or task where you used {skill_name}? "
            "What was your contribution and what result did you achieve?"
        )

    if missing_skills and not required_missing:
        for skill_name in missing_skills[:2]:
            questions.append(
                f"Your resume does not explicitly mention {skill_name}. "
                "Do you have relevant exposure through work, projects, coursework, or self-learning?"
            )

    if sections.get("projects"):
        questions.append(
            "Pick your strongest project and walk me through the architecture, "
            "your individual contribution, the hardest problem, and how you measured success."
        )

    questions.append(
        "Describe a technical decision that required a trade-off. "
        "What alternatives did you consider?"
    )

    if experience > 0:
        questions.append(
            "Which achievement best demonstrates your readiness for this role, "
            "and how did you measure its impact?"
        )

    questions = _unique(questions)

    evidence.extend([
        f"ATS score: {ats:.1f}%",
        f"Semantic match: {semantic:.1f}%",
        f"Skill match: {skill:.1f}%",
        f"Lexical match: {lexical:.1f}%",
        f"Keyword match: {keyword:.1f}%",
        f"Resume quality: {quality_score:.1f}%",
        f"Detected technical skills: {len(all_skills)}",
        f"Section completeness: {section_completeness:.1f}%",
        f"Contact completeness: {contact_completeness:.1f}%",
    ])

    if requirement is not None:
        evidence.append(f"Requirement match: {requirement:.1f}%")

    if required_matched or required_missing:
        evidence.append(
            f"Required skills matched: {len(required_matched)}; "
            f"missing: {len(required_missing)}"
        )

    summary = (
        f"{candidate} has an ATS match of {ats:.1f}%"
        + (
            f" and requirement coverage of {requirement:.1f}%"
            if requirement is not None
            else ""
        )
        + ". This brief summarizes existing evidence, gaps, and interview verification areas."
    )

    return {
        "candidate": candidate,
        "summary": summary,
        "strengths": strengths,
        "gaps": gaps,
        "interview_focus": focus,
        "talking_points": talking_points,
        "interview_questions": questions,
        "evidence": evidence,
        "metrics": {
            "ats_score": round(ats, 1),
            "requirement_score": round(requirement, 1) if requirement is not None else None,
            "quality_score": round(quality_score, 1),
            "semantic_score": round(semantic, 1),
            "skill_score": round(skill, 1),
            "lexical_score": round(lexical, 1),
            "keyword_score": round(keyword, 1),
            "experience_years": round(experience, 1),
            "skills_detected": len(all_skills),
        },
        "generated_from": [
            "Hybrid ATS analysis",
            "Requirement matching",
            "Resume intelligence",
            "Resume quality analysis",
            "Semantic/lexical/keyword matching",
        ],
    }


def brief_to_text(brief: Dict[str, Any]) -> str:
    lines = [
        "AI-ASSISTED CANDIDATE SCREENING BRIEF",
        "=" * 50,
        f"Candidate: {brief.get('candidate', 'Candidate')}",
        "",
        "SCREENING SUMMARY",
        "-" * 20,
        brief.get("summary", ""),
        "",
        "KEY METRICS",
        "-" * 20,
    ]

    metrics = brief.get("metrics", {}) or {}

    metric_defs = [
        ("ATS Score", "ats_score", "%"),
        ("Requirement Match", "requirement_score", "%"),
        ("Resume Quality", "quality_score", "%"),
        ("Semantic Match", "semantic_score", "%"),
        ("Skill Match", "skill_score", "%"),
        ("Lexical Match", "lexical_score", "%"),
        ("Keyword Match", "keyword_score", "%"),
        ("Experience", "experience_years", " years"),
        ("Technical Skills", "skills_detected", ""),
    ]

    for label, key, suffix in metric_defs:
        value = metrics.get(key)
        if value is None:
            display = "Not available"
        elif isinstance(value, float):
            display = f"{value:.1f}{suffix}"
        else:
            display = f"{value}{suffix}"
        lines.append(f"{label}: {display}")

    for title, key in [
        ("KEY STRENGTHS", "strengths"),
        ("KEY GAPS / VERIFICATION AREAS", "gaps"),
        ("INTERVIEW FOCUS AREAS", "interview_focus"),
        ("RECRUITER TALKING POINTS", "talking_points"),
        ("SUGGESTED INTERVIEW QUESTIONS", "interview_questions"),
        ("EVIDENCE", "evidence"),
    ]:
        lines.extend(["", title, "-" * 20])
        values = _list(brief.get(key, []))
        if values:
            lines.extend(f"- {x}" for x in values)
        else:
            lines.append("None identified.")

    lines.extend([
        "",
        "Generated from existing platform analysis signals; "
        "use the brief as a recruiter aid and verify it against the resume "
        "and interview evidence."
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    brief = build_screening_brief(
        candidate_row={"Candidate": "Adhyan Sharma", "ATS Score": 84},
        analysis_result={
            "matched_skills": ["Python", "Machine Learning"],
            "missing_skills": ["Docker", "AWS"],
            "matched_keywords": ["NLP", "FastAPI"],
        },
        ats_result={
            "ats_score": 84,
            "semantic_score": 79,
            "skill_score": 86,
            "lexical_score": 71,
            "keyword_score": 80,
        },
        intelligence={
            "experience_years": 1.5,
            "all_skills": ["Python", "Machine Learning", "NLP"],
            "sections": {"projects": "Project A"},
            "section_completeness": 90,
            "contact_completeness": 100,
        },
        quality={
            "quality_score": 88,
            "quantification": {"score": 65},
            "repetition": {"score": 80},
        },
        requirements={
            "requirement_match_score": 81,
            "required_skills": {"matched": ["Python"], "missing": ["AWS"]},
            "preferred_skills": {"matched": ["NLP"], "missing": []},
            "experience": {"required_years": 1, "candidate_years": 1.5, "status": "Meets requirement"},
            "education": {"status": "Meets requirement"},
        },
    )

    text = brief_to_text(brief)
    assert "Adhyan Sharma" in text
    assert "AWS" in text
    assert "INTERVIEW FOCUS AREAS" in text
    print("AI-assisted screening brief engine is working correctly.")
