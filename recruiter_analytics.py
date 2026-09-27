
"""
Recruiter Analytics V8

Portfolio-friendly analytics layer for the AI Resume Screening Platform.

The module provides descriptive recruiter analytics across:
- pipeline status
- ATS / semantic / skill / quality metrics
- required-skill coverage
- common missing required skills
- interview status
- communication activity

No new score is created and no hiring recommendation is produced.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


PIPELINE_STATUSES = [
    "New",
    "Reviewing",
    "Shortlisted",
    "Hold",
    "Rejected",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []

    for value in values:
        item = _clean(value)
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def pipeline_status_counts(
    df: pd.DataFrame,
    statuses: Sequence[str] = PIPELINE_STATUSES,
) -> pd.DataFrame:
    counts = (
        df["Status"]
        .astype(str)
        .value_counts()
        if df is not None and "Status" in df.columns
        else pd.Series(dtype="int64")
    )

    rows = [
        {
            "Status": status,
            "Candidates": int(counts.get(status, 0)),
        }
        for status in statuses
    ]

    return pd.DataFrame(rows)


def score_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Average",
                "Minimum",
                "Maximum",
            ]
        )

    metrics = [
        "ATS Score",
        "Requirement Match",
        "Semantic Score",
        "Skill Score",
        "Keyword Score",
        "Resume Quality",
    ]

    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = pd.to_numeric(
            df[metric],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "Metric": metric,
                "Average": round(float(values.mean()), 1),
                "Minimum": round(float(values.min()), 1),
                "Maximum": round(float(values.max()), 1),
            }
        )

    return pd.DataFrame(rows)


def shortlisted_summary(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    total = int(len(df)) if df is not None else 0

    if total == 0:
        return {
            "total": 0,
            "shortlisted": 0,
            "shortlist_rate": 0.0,
        }

    shortlisted = int(
        df["Shortlisted"].sum()
    ) if "Shortlisted" in df.columns else 0

    return {
        "total": total,
        "shortlisted": shortlisted,
        "shortlist_rate": round(
            (shortlisted / total) * 100,
            1,
        ),
    }


def required_skill_gap_frequency(
    resume_cache: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    counter = Counter()

    for cache in (resume_cache or {}).values():
        requirements = (
            cache.get(
                "requirements",
                {},
            )
            or {}
        )

        required = (
            requirements.get(
                "required_skills",
                {},
            )
            or {}
        )

        for skill in required.get(
            "missing",
            [],
        ) or []:
            skill = _clean(skill)
            if skill:
                counter[skill] += 1

    # Convert the frequency counter into a table.
    normalized_rows = []
    for skill, count in counter.most_common():
        normalized_rows.append(
            {
                "Required Skill": skill,
                "Candidates Missing": int(count),
            }
        )

    return pd.DataFrame(
        normalized_rows,
        columns=[
            "Required Skill",
            "Candidates Missing",
        ],
    )


def preferred_skill_gap_frequency(
    resume_cache: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    counter = Counter()

    for cache in (resume_cache or {}).values():
        requirements = (
            cache.get(
                "requirements",
                {},
            )
            or {}
        )

        preferred = (
            requirements.get(
                "preferred_skills",
                {},
            )
            or {}
        )

        for skill in preferred.get(
            "missing",
            [],
        ) or []:
            skill = _clean(skill)
            if skill:
                counter[skill] += 1

    return pd.DataFrame(
        [
            {
                "Preferred Skill": skill,
                "Candidates Missing": int(count),
            }
            for skill, count in counter.most_common()
        ],
        columns=[
            "Preferred Skill",
            "Candidates Missing",
        ],
    )


def required_skill_coverage_table(
    df: pd.DataFrame,
    resume_cache: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    skill_sets = []

    # Prefer the union from existing deterministic requirement matching.
    for cache in (resume_cache or {}).values():
        requirements = (
            cache.get(
                "requirements",
                {},
            )
            or {}
        )

        required = (
            requirements.get(
                "required_skills",
                {},
            )
            or {}
        )

        for skill in (
            required.get("matched", [])
            or []
        ):
            skill_sets.append(_clean(skill))

        for skill in (
            required.get("missing", [])
            or []
        ):
            skill_sets.append(_clean(skill))

    skills = _unique(skill_sets)

    if not skills or df is None or df.empty:
        return pd.DataFrame()

    candidate_lookup = {}

    for _, row in df.iterrows():
        resume_name = _clean(
            row.get(
                "Resume",
                "",
            )
        )

        if resume_name:
            candidate_lookup[
                resume_name
            ] = _clean(
                row.get(
                    "Candidate",
                    resume_name,
                )
            )

    rows = []

    for resume_name, cache in (
        resume_cache or {}
    ).items():
        requirements = (
            cache.get(
                "requirements",
                {},
            )
            or {}
        )

        required = (
            requirements.get(
                "required_skills",
                {},
            )
            or {}
        )

        matched = {
            _clean(skill).lower()
            for skill in (
                required.get("matched", [])
                or []
            )
        }

        missing = {
            _clean(skill).lower()
            for skill in (
                required.get("missing", [])
                or []
            )
        }

        row = {
            "Candidate": candidate_lookup.get(
                resume_name,
                resume_name,
            )
        }

        for skill in skills:
            key = skill.lower()

            if key in matched:
                row[skill] = "Matched"
            elif key in missing:
                row[skill] = "Missing"
            else:
                row[skill] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


def interview_activity_summary(
    interviews: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    summary = {
        "total": 0,
        "scheduled": 0,
        "completed": 0,
        "cancelled": 0,
        "rescheduled": 0,
        "pending_feedback": 0,
    }

    for interview in interviews or []:
        summary["total"] += 1

        status = _clean(
            interview.get(
                "status",
                "",
            )
        ).lower()

        if status == "scheduled":
            summary["scheduled"] += 1
        elif status == "completed":
            summary["completed"] += 1
        elif status == "cancelled":
            summary["cancelled"] += 1
        elif status == "rescheduled":
            summary["rescheduled"] += 1

        if (
            status == "completed"
            and not _clean(
                interview.get(
                    "feedback",
                    "",
                )
            )
        ):
            summary["pending_feedback"] += 1

    return summary


def communication_activity_summary(
    messages: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    counter = Counter()

    for message in messages or []:
        label = _clean(
            message.get(
                "communication_type",
                "Unknown",
            )
        ) or "Unknown"

        counter[label] += 1

    return pd.DataFrame(
        [
            {
                "Communication Type": label,
                "Messages": int(count),
            }
            for label, count in counter.most_common()
        ],
        columns=[
            "Communication Type",
            "Messages",
        ],
    )


def build_analytics_snapshot(
    *,
    df: pd.DataFrame,
    resume_cache: Dict[str, Dict[str, Any]],
    interviews: Sequence[Dict[str, Any]] = (),
    communications: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    scores = score_statistics(df)
    shortlist = shortlisted_summary(df)

    interview_summary = interview_activity_summary(
        interviews
    )

    return {
        "candidate_count": int(
            len(df)
            if df is not None
            else 0
        ),
        "shortlisted": shortlist,
        "pipeline": pipeline_status_counts(df),
        "scores": scores,
        "required_skill_gaps": required_skill_gap_frequency(
            resume_cache
        ),
        "preferred_skill_gaps": preferred_skill_gap_frequency(
            resume_cache
        ),
        "required_skill_coverage": required_skill_coverage_table(
            df,
            resume_cache,
        ),
        "interviews": interview_summary,
        "communications": communication_activity_summary(
            communications
        ),
    }
