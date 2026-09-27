"""
Recruiter Kanban Pipeline
-------------------------
Pure helper functions for the Streamlit recruiter pipeline board.

The board uses the same persistent workflow states already used by the app:
New -> Reviewing -> Shortlisted -> Hold -> Rejected
"""

from typing import Dict, List, Tuple


KANBAN_STATUSES = [
    "New",
    "Reviewing",
    "Shortlisted",
    "Hold",
    "Rejected",
]

STATUS_DESCRIPTIONS = {
    "New": "Newly screened candidates",
    "Reviewing": "Candidates under active review",
    "Shortlisted": "Candidates selected for the next stage",
    "Hold": "Candidates temporarily paused",
    "Rejected": "Candidates not moving forward",
}


def normalize_number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_status_index(status: str) -> int:
    try:
        return KANBAN_STATUSES.index(status)
    except ValueError:
        return 0


def move_status(status: str, direction: int) -> str:
    """Move one step left/right in the pipeline."""
    current_index = get_status_index(status)
    new_index = max(
        0,
        min(
            len(KANBAN_STATUSES) - 1,
            current_index + int(direction),
        ),
    )
    return KANBAN_STATUSES[new_index]


def status_counts(candidates: List[Dict], workflow: Dict) -> Dict[str, int]:
    """Count candidates in each pipeline stage."""
    counts = {status: 0 for status in KANBAN_STATUSES}

    for candidate in candidates:
        resume_name = str(candidate.get("Resume", ""))
        state = workflow.get(
            resume_name,
            {
                "status": "New",
                "shortlisted": False,
                "notes": "",
            },
        )
        status = state.get("status", "New")
        if status not in counts:
            status = "New"
        counts[status] += 1

    return counts


def prepare_board_candidates(
    candidates: List[Dict],
    workflow: Dict,
    search_text: str = "",
    min_ats: float = 0.0,
    min_requirement: float = 0.0,
    shortlisted_only: bool = False,
    sort_by: str = "ATS Score",
) -> List[Dict]:
    """Filter and enrich candidates for kanban rendering."""
    search = str(search_text or "").strip().lower()
    prepared: List[Dict] = []

    for candidate in candidates:
        candidate_name = str(candidate.get("Candidate", "Candidate"))
        resume_name = str(candidate.get("Resume", ""))

        if search:
            haystack = f"{candidate_name} {resume_name}".lower()
            if search not in haystack:
                continue

        ats_score = normalize_number(candidate.get("ATS Score", 0))
        requirement_score = normalize_number(
            candidate.get("Requirement Match", 0)
        )

        if ats_score < float(min_ats):
            continue

        if requirement_score < float(min_requirement):
            continue

        state = workflow.get(
            resume_name,
            {
                "status": "New",
                "shortlisted": False,
                "notes": "",
            },
        )

        shortlisted = bool(state.get("shortlisted", False))

        if shortlisted_only and not shortlisted:
            continue

        enriched = dict(candidate)
        enriched["workflow_status"] = state.get("status", "New")
        enriched["workflow_shortlisted"] = shortlisted
        enriched["workflow_notes"] = str(state.get("notes", ""))
        prepared.append(enriched)

    reverse = True
    if sort_by == "Candidate":
        reverse = False
        prepared.sort(key=lambda x: str(x.get("Candidate", "")).lower())
    else:
        prepared.sort(
            key=lambda x: normalize_number(x.get(sort_by, 0)),
            reverse=reverse,
        )

    return prepared


def board_summary(candidates: List[Dict], workflow: Dict) -> Dict[str, int]:
    counts = status_counts(candidates, workflow)
    total = len(candidates)
    shortlisted = sum(
        1
        for candidate in candidates
        if bool(
            workflow.get(
                str(candidate.get("Resume", "")),
                {},
            ).get("shortlisted", False)
        )
    )

    return {
        "total": total,
        "shortlisted": shortlisted,
        "new": counts["New"],
        "reviewing": counts["Reviewing"],
        "hold": counts["Hold"],
        "rejected": counts["Rejected"],
    }


def split_into_columns(
    candidates: List[Dict],
    statuses: List[str] = None,
) -> Dict[str, List[Dict]]:
    """Group prepared candidates by workflow status."""
    if statuses is None:
        statuses = KANBAN_STATUSES

    columns = {status: [] for status in statuses}

    for candidate in candidates:
        status = candidate.get("workflow_status", "New")
        if status not in columns:
            status = "New"
        columns[status].append(candidate)

    return columns


def candidate_card_metrics(candidate: Dict) -> Tuple[float, float, float, int]:
    """Return the four most useful card metrics."""
    ats = normalize_number(candidate.get("ATS Score", 0))
    requirement = normalize_number(candidate.get("Requirement Match", 0))
    quality = normalize_number(candidate.get("Resume Quality", 0))
    experience = normalize_number(candidate.get("Experience Years", 0))
    return ats, requirement, quality, int(round(experience))


if __name__ == "__main__":
    sample = [
        {"Candidate": "A", "Resume": "a.pdf", "ATS Score": 80},
        {"Candidate": "B", "Resume": "b.pdf", "ATS Score": 90},
    ]
    workflow = {
        "a.pdf": {"status": "Reviewing", "shortlisted": False, "notes": ""},
        "b.pdf": {"status": "Shortlisted", "shortlisted": True, "notes": ""},
    }

    prepared = prepare_board_candidates(sample, workflow)
    columns = split_into_columns(prepared)

    assert move_status("New", 1) == "Reviewing"
    assert move_status("Rejected", 1) == "Rejected"
    assert len(columns["Reviewing"]) == 1
    assert len(columns["Shortlisted"]) == 1

    print("Recruiter Kanban engine is working correctly.")
