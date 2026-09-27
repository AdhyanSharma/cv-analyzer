"""Recruiter workflow state and filtering helpers."""

from typing import Dict, List, Optional


STATUS_OPTIONS = [
    "New",
    "Reviewing",
    "Shortlisted",
    "Hold",
    "Rejected",
]

DEFAULT_STATUS = "New"


def create_candidate_state() -> Dict:
    return {
        "status": DEFAULT_STATUS,
        "shortlisted": False,
        "notes": "",
    }


def initialize_workflow(
    candidate_names: List[str],
    existing_workflow: Optional[Dict] = None,
) -> Dict:
    workflow = dict(existing_workflow or {})

    for candidate in candidate_names:
        if not candidate:
            continue

        if candidate not in workflow:
            workflow[candidate] = create_candidate_state()
        else:
            workflow[candidate].setdefault("status", DEFAULT_STATUS)
            workflow[candidate].setdefault("shortlisted", False)
            workflow[candidate].setdefault("notes", "")

    return workflow


def get_candidate_state(workflow: Dict, candidate_name: str) -> Dict:
    if candidate_name not in workflow:
        workflow[candidate_name] = create_candidate_state()
    return workflow[candidate_name]


def set_candidate_status(workflow: Dict, candidate_name: str, status: str) -> Dict:
    if status not in STATUS_OPTIONS:
        raise ValueError(
            f"Invalid status '{status}'. Allowed statuses: {STATUS_OPTIONS}"
        )

    state = get_candidate_state(workflow, candidate_name)
    state["status"] = status

    if status == "Shortlisted":
        state["shortlisted"] = True
    elif status == "Rejected":
        state["shortlisted"] = False

    return workflow


def set_shortlist(workflow: Dict, candidate_name: str, shortlisted: bool) -> Dict:
    state = get_candidate_state(workflow, candidate_name)
    state["shortlisted"] = bool(shortlisted)

    if shortlisted:
        state["status"] = "Shortlisted"
    elif state["status"] == "Shortlisted":
        state["status"] = "Reviewing"

    return workflow


def set_candidate_notes(workflow: Dict, candidate_name: str, notes: str) -> Dict:
    state = get_candidate_state(workflow, candidate_name)
    state["notes"] = str(notes or "").strip()
    return workflow


def filter_candidates(
    candidates: List[Dict],
    workflow: Dict,
    search_text: str = "",
    status_filter: str = "All",
    shortlisted_only: bool = False,
    min_ats_score: float = 0.0,
    min_requirement_score: float = 0.0,
) -> List[Dict]:
    search = str(search_text or "").strip().lower()
    filtered = []

    for candidate in candidates:
        candidate_name = str(candidate.get("Candidate", ""))
        resume_name = str(candidate.get("Resume", ""))

        if search:
            haystack = f"{candidate_name} {resume_name}".lower()
            if search not in haystack:
                continue

        state = workflow.get(resume_name, create_candidate_state())
        status = state.get("status", DEFAULT_STATUS)
        shortlisted = bool(state.get("shortlisted", False))

        if status_filter != "All" and status != status_filter:
            continue

        if shortlisted_only and not shortlisted:
            continue

        try:
            ats = float(candidate.get("ATS Score", 0) or 0)
        except (TypeError, ValueError):
            ats = 0.0

        try:
            requirement = float(candidate.get("Requirement Match", 0) or 0)
        except (TypeError, ValueError):
            requirement = 0.0

        if ats < min_ats_score:
            continue

        if requirement < min_requirement_score:
            continue

        filtered.append(candidate)

    return filtered


def calculate_workflow_summary(candidates: List[Dict], workflow: Dict) -> Dict:
    summary = {
        "total": len(candidates),
        "new": 0,
        "reviewing": 0,
        "shortlisted": 0,
        "hold": 0,
        "rejected": 0,
    }

    for candidate in candidates:
        resume_name = str(candidate.get("Resume", ""))
        state = workflow.get(resume_name, create_candidate_state())
        status = state.get("status", DEFAULT_STATUS)

        if state.get("shortlisted", False):
            summary["shortlisted"] += 1

        if status == "New":
            summary["new"] += 1
        elif status == "Reviewing":
            summary["reviewing"] += 1
        elif status == "Hold":
            summary["hold"] += 1
        elif status == "Rejected":
            summary["rejected"] += 1

    return summary


def calculate_shortlist_rate(candidates: List[Dict], workflow: Dict) -> float:
    total = len(candidates)
    if total == 0:
        return 0.0

    summary = calculate_workflow_summary(candidates, workflow)
    return (summary["shortlisted"] / total) * 100.0


def validate_workflow(workflow: Dict) -> bool:
    if not isinstance(workflow, dict):
        return False

    for state in workflow.values():
        if not isinstance(state, dict):
            return False
        if state.get("status", DEFAULT_STATUS) not in STATUS_OPTIONS:
            return False
        if not isinstance(state.get("shortlisted", False), bool):
            return False
        if not isinstance(state.get("notes", ""), str):
            return False

    return True


if __name__ == "__main__":
    sample = [
        {"Candidate": "Rahul Sharma", "Resume": "Rahul.pdf", "ATS Score": 86, "Requirement Match": 82},
        {"Candidate": "Priya Singh", "Resume": "Priya.pdf", "ATS Score": 74, "Requirement Match": 69},
        {"Candidate": "Aman Verma", "Resume": "Aman.pdf", "ATS Score": 91, "Requirement Match": 88},
    ]

    workflow = initialize_workflow([row["Resume"] for row in sample])
    set_candidate_status(workflow, "Rahul.pdf", "Reviewing")
    set_shortlist(workflow, "Aman.pdf", True)
    set_candidate_notes(workflow, "Aman.pdf", "Review project depth.")

    print(calculate_workflow_summary(sample, workflow))
    print(calculate_shortlist_rate(sample, workflow))
    assert validate_workflow(workflow)
    print("Recruiter workflow engine is working correctly.")
