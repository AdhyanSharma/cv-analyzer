"""
AI Resume Screening Platform
Main Streamlit Application

This version focuses on:
    - Stable candidate extraction
    - Correct names
    - Email / phone visibility
    - LinkedIn / GitHub extraction and clickable links
    - DOCX/PDF hyperlink recovery through resume_utils.py
    - Hybrid ATS analysis
    - Resume quality
    - Requirement matching
    - Cleaner recruiter dashboard
"""

import os
import re
import hashlib
import zipfile
from io import BytesIO

import pandas as pd
import streamlit as st
from fpdf import FPDF

from resume_utils import extract_resume_text
from analysis import analyze_resumes_batch
from ats_engine import analyze_ats
from resume_intelligence import analyze_resume_intelligence
from resume_quality import analyze_resume_quality
from jd_intelligence_v6 import (
    analyze_job_description,
    extract_top_jd_keywords,
    build_rag_gemini_jd_intelligence,
)
from matching_engine import analyze_candidate_requirements
from jd_requirement_matrix import (
    build_requirement_matrix,
    requirements_to_dataframe,
    build_candidate_requirement_matrix,
    build_requirement_summary,
    requirement_matrix_to_text,
)
from recruiter_analytics import (
    build_analytics_snapshot,
)
from recruiter_workflow import (
    STATUS_OPTIONS,
    initialize_workflow,
    set_candidate_status,
    set_shortlist,
    set_candidate_notes,
    filter_candidates,
    calculate_workflow_summary,
    calculate_shortlist_rate,
)
from candidate_profile import (
    build_candidate_snapshot,
    build_screening_summary,
    join_values,
    safe_float,
)
from recruiter_store import (
    DB_PATH,
    initialize_database,
    load_workflow,
    upsert_candidate,
    save_candidate_state,
    get_status_history,
    get_note_history,
)
from recruiter_kanban import (
    KANBAN_STATUSES,
    STATUS_DESCRIPTIONS,
    move_status,
    prepare_board_candidates,
    split_into_columns,
    board_summary,
)
from interview_manager import (
    INTERVIEW_ROUNDS,
    INTERVIEW_TYPES,
    INTERVIEW_STATUSES,
    RECOMMENDATION_OPTIONS,
    initialize_interview_database,
    create_interview,
    get_interview,
    update_interview,
    list_interviews,
    get_candidate_interview_history,
    get_interview_event_history,
    get_interview_statistics,
    generate_ics_event,
)
from recruiter_communication import (
    TEMPLATE_OPTIONS,
    initialize_communication_database,
    generate_communication,
    save_communication,
    list_communications,
    build_mailto_url,
)
from screening_brief import (
    build_screening_brief,
    brief_to_text,
)
from gemini_screening_brief import (
    DEFAULT_MODEL as GEMINI_MODEL,
    is_configured as gemini_is_configured,
    generate_gemini_screening_brief,
    gemini_brief_to_text,
)
from rag_recruiter_copilot import (
    DEFAULT_GEMINI_MODEL as GEMINI_COPILOT_MODEL,
    build_copilot_corpus,
    retrieve_evidence,
    generate_gemini_copilot_answer,
    evidence_only_fallback,
)
from candidate_comparison import (
    build_candidate_comparison,
    comparison_to_text,
    build_comparison_evidence,
)
from explainability_engine import (
    build_explainability,
    explanation_to_text,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "analysis_results": None,
    "resume_cache": {},
    "pdf_reports": {},
    "jd_text": "",
    "jd_keywords": [],
    "jd_keyword_groups": {},
    "jd_keyword_retrieved": [],
    "jd_keyword_source": "Local hybrid NLP",
    "jd_keyword_warning": "",
    "jd_analysis": {},
    "jd_rag_gemini": {},
    "jd_rag_gemini_fingerprint": "",
    "jd_rag_gemini_warning": "",
    "jd_requirement_matrix": [],
    "candidate_workflow": {},
    "selected_candidate_resume": "",
    "kanban_selected_candidate": "",
    "interview_selected_id": None,
    "communication_selected_candidate": "",
    "screening_brief": {},
    "gemini_screening_brief": {},
    "copilot_history": {},
    "comparison_history": {},
    "job_fingerprint": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize persistent recruiter storage once per app process.
initialize_database()
initialize_interview_database()
initialize_communication_database()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.03rem;
        margin-bottom: 20px;
    }

    .profile-card {
        border: 1px solid #30343b;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(filename: str) -> str:
    if not filename:
        return "candidate"

    stem = os.path.splitext(
        os.path.basename(str(filename))
    )[0]

    stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem,
    )

    stem = re.sub(
        r"_+",
        "_",
        stem,
    )

    stem = stem.strip("_.-")

    return stem or "candidate"


def build_job_fingerprint(jd_text: str) -> str:
    """Create a stable identifier for one Job Description."""
    normalized = re.sub(r"\s+", " ", str(jd_text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def persist_workflow_state(
    resume_name: str,
    state: dict,
) -> None:
    """Persist the current recruiter workflow state for one candidate."""
    job_fingerprint = st.session_state.get("job_fingerprint", "")
    if not job_fingerprint or not resume_name:
        return

    save_candidate_state(
        job_fingerprint=job_fingerprint,
        resume_name=str(resume_name),
        status=str(state.get("status", "New")),
        shortlisted=bool(state.get("shortlisted", False)),
        notes=str(state.get("notes", "")),
    )


def sync_workflow_from_database(df: pd.DataFrame) -> dict:
    """Load persisted recruiter state for the current JD/candidate set."""
    job_fingerprint = st.session_state.get("job_fingerprint", "")
    if not job_fingerprint or df is None or df.empty:
        return {}

    resume_names = df["Resume"].astype(str).tolist()
    stored = load_workflow(job_fingerprint, resume_names)

    # Preserve any in-memory candidates that are not yet represented in DB.
    existing = st.session_state.get("candidate_workflow", {})
    merged = dict(existing)
    merged.update(stored)
    return merged


# ============================================================
# PDF HELPERS
# ============================================================

def safe_pdf_text(text) -> str:
    if text is None:
        return ""

    text = str(text)

    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "•": "-",
        "→": "->",
        "✓": "[OK]",
        "✗": "[X]",
        "⚠": "[WARNING]",
        "₹": "INR ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return (
        text
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


def split_long_words(
    text: str,
    max_length: int = 25,
) -> str:
    if not text:
        return ""

    safe_words = []

    for word in text.split(" "):
        if len(word) <= max_length:
            safe_words.append(word)
        else:
            chunks = [
                word[i:i + max_length]
                for i in range(0, len(word), max_length)
            ]
            safe_words.append(" ".join(chunks))

    return " ".join(safe_words)


def safe_multi_cell(
    pdf,
    text,
    height=7,
):
    text = safe_pdf_text(text)
    text = split_long_words(text)

    if not text:
        text = " "

    try:
        pdf.multi_cell(0, height, text)
    except Exception:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0,
            height,
            "Text could not be rendered.",
        )


def add_pdf_section(
    pdf,
    title,
    values,
):
    pdf.set_font("Arial", "B", 12)
    safe_multi_cell(pdf, title, 7)

    pdf.set_font("Arial", "", 10)

    if not values:
        safe_multi_cell(pdf, "None detected.", 6)
        return

    if isinstance(values, str):
        values = [values]

    for value in values:
        if value is None:
            continue

        safe_multi_cell(
            pdf,
            f"- {value}",
            6,
        )


def generate_pdf_report(
    candidate_name,
    resume_name,
    ats,
    intelligence,
    quality,
    requirements,
    analysis_result,
):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    safe_multi_cell(
        pdf,
        "AI Resume Screening Report",
        10,
    )

    pdf.set_font("Arial", "", 10)
    safe_multi_cell(
        pdf,
        f"Candidate: {candidate_name}",
        6,
    )
    safe_multi_cell(
        pdf,
        f"Resume: {resume_name}",
        6,
    )

    pdf.ln(4)

    add_pdf_section(
        pdf,
        "Overall Scores",
        [
            f"ATS Match: {ats.get('ats_score', 0)}%",
            f"Resume Quality: {quality.get('quality_score', 0)}%",
            f"Requirement Match: {requirements.get('requirement_match_score') or 'N/A'}%",
            f"Skill Match: {ats.get('skill_score', 0)}%",
            f"Semantic Match: {ats.get('semantic_score', 0)}%",
            f"Lexical Match: {ats.get('lexical_score', 0)}%",
            f"Keyword Match: {ats.get('keyword_score', 0)}%",
        ],
    )

    add_pdf_section(
        pdf,
        "Candidate Contact",
        [
            f"Email: {intelligence.get('email') or 'Not detected'}",
            f"Phone: {intelligence.get('phone') or 'Not detected'}",
            f"LinkedIn: {intelligence.get('linkedin') or 'Not detected'}",
            f"GitHub: {intelligence.get('github') or 'Not detected'}",
        ],
    )

    add_pdf_section(
        pdf,
        "Education",
        intelligence.get("education", ""),
    )

    add_pdf_section(
        pdf,
        "Experience",
        [
            f"Explicitly stated experience: "
            f"{intelligence.get('experience_years', 0)} years"
        ],
    )

    add_pdf_section(
        pdf,
        "Detected Skills",
        intelligence.get("all_skills", []),
    )

    add_pdf_section(
        pdf,
        "Matched Skills",
        analysis_result.get("matched_skills", []),
    )

    add_pdf_section(
        pdf,
        "Missing Skills",
        analysis_result.get("missing_skills", []),
    )

    required = requirements.get("required_skills", {})
    preferred = requirements.get("preferred_skills", {})

    add_pdf_section(
        pdf,
        "Matched Required Skills",
        required.get("matched", []),
    )

    add_pdf_section(
        pdf,
        "Missing Required Skills",
        required.get("missing", []),
    )

    add_pdf_section(
        pdf,
        "Matched Preferred Skills",
        preferred.get("matched", []),
    )

    add_pdf_section(
        pdf,
        "Projects",
        intelligence.get("projects", ""),
    )

    add_pdf_section(
        pdf,
        "Certifications",
        intelligence.get("certifications", ""),
    )

    add_pdf_section(
        pdf,
        "Resume Quality Recommendations",
        quality.get("recommendations", []),
    )

    output = pdf.output(dest="S")

    if isinstance(output, str):
        pdf_bytes = output.encode("latin-1")
    else:
        pdf_bytes = bytes(output)

    return BytesIO(pdf_bytes)


# ============================================================
# EXPORT HELPERS
# ============================================================

def create_excel_report(dataframe: pd.DataFrame) -> BytesIO:
    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Candidate Ranking",
        )

    output.seek(0)
    return output


def create_zip_reports(pdf_reports) -> BytesIO:
    output = BytesIO()

    with zipfile.ZipFile(
        output,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename, buffer in pdf_reports.items():
            archive.writestr(
                filename,
                buffer.getvalue(),
            )

    output.seek(0)
    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    top_n_keywords = st.slider(
        "JD Keywords",
        min_value=10,
        max_value=40,
        value=20,
    )

    use_jd_rag = st.checkbox(
        "🧠 Run RAG + Gemini JD Intelligence",
        value=False,
        help=(
            "Retrieve focused JD evidence and optionally use Gemini to "
            "add grounded role, domain, concept and responsibility "
            "intelligence. Deterministic extraction remains authoritative."
        ),
    )

    minimum_ats = st.slider(
        "Minimum ATS Filter",
        min_value=0,
        max_value=100,
        value=0,
    )

    st.divider()

    st.markdown(
        """
        ### ATS Scoring

        Skill Match — **35%**

        Semantic Match — **30%**

        Lexical Match — **20%**

        Keyword Match — **15%**

        ---

        Resume Quality is a separate analysis from
        role relevance.
        """
    )

    st.divider()
    st.caption("💾 Persistent recruiter storage: SQLite")
    st.caption(f"Database: {os.path.basename(DB_PATH)}")

    st.divider()
    st.caption(
        "✨ Free Gemini Screening: "
        + (
            "Configured"
            if gemini_is_configured()
            else "Not configured"
        )
    )
    st.caption(f"Gemini model: {GEMINI_MODEL} (free-tier fallback enabled)")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Resume Screening Platform</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Hybrid NLP screening with candidate intelligence,
    requirement matching and resume quality analysis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOADS
# ============================================================

upload1, upload2 = st.columns(2)

with upload1:
    st.subheader("📄 Job Description")

    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx"],
        key="jd_upload",
    )

with upload2:
    st.subheader("👥 Candidate Resumes")

    resume_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_upload",
    )


# ============================================================
# ANALYZE
# ============================================================

if jd_file and resume_files:
    st.divider()

    st.info(
        f"{len(resume_files)} resume(s) ready for analysis."
    )

    if st.button(
        "🚀 Analyze Resumes",
        type="primary",
        use_container_width=True,
    ):
        # Clear previous results.
        for key, value in {
            "analysis_results": None,
            "resume_cache": {},
            "pdf_reports": {},
            "jd_text": "",
            "jd_keywords": [],
            "jd_keyword_groups": {},
            "jd_keyword_retrieved": [],
            "jd_keyword_source": "Local hybrid NLP",
            "jd_keyword_warning": "",
            "jd_analysis": {},
            "jd_rag_gemini": {},
            "jd_rag_gemini_fingerprint": "",
            "jd_rag_gemini_warning": "",
            "jd_requirement_matrix": [],
            "candidate_workflow": {},
            "selected_candidate_resume": "",
            "kanban_selected_candidate": "",
            "screening_brief": {},
            "gemini_screening_brief": {},
            "copilot_history": {},
            "comparison_history": {},
            "job_fingerprint": "",
        }.items():
            st.session_state[key] = value

        # ----------------------------------------------------
        # JD
        # ----------------------------------------------------
        with st.spinner("Reading Job Description..."):
            jd_text = extract_resume_text(jd_file)

        if not jd_text:
            st.error(
                "Could not extract text from the Job Description."
            )
            st.stop()

        jd_analysis = analyze_job_description(jd_text)

        # ----------------------------------------------------
        # Hybrid / section-aware JD keyword extraction
        # ----------------------------------------------------
        jd_keywords = extract_top_jd_keywords(
            jd_text,
            n=top_n_keywords,
        )

        # ----------------------------------------------------
        # Optional RAG + Gemini JD Intelligence V4
        #
        # Cache by JD fingerprint so normal Streamlit reruns do not
        # repeatedly consume Gemini quota for the same JD.
        # ----------------------------------------------------
        current_jd_fingerprint = build_job_fingerprint(
            jd_text
        )

        rag_gemini_result = {}
        if (
            use_jd_rag
            and (
                st.session_state.get(
                    "jd_rag_gemini_fingerprint",
                    "",
                )
                != current_jd_fingerprint
            )
        ):
            with st.spinner(
                "🔎 Retrieving JD evidence and generating structured JD intelligence..."
            ):
                rag_gemini_result = build_rag_gemini_jd_intelligence(
                    jd_text,
                    top_k_per_query=3,
                )

            st.session_state.jd_rag_gemini = (
                rag_gemini_result
            )

            st.session_state.jd_rag_gemini_fingerprint = (
                current_jd_fingerprint
            )

        elif use_jd_rag:
            rag_gemini_result = st.session_state.get(
                "jd_rag_gemini",
                {},
            )

        # Keyword groups come from the deterministic V4 analysis.
        # RAG/Gemini is additive and does not replace this base signal.
        keyword_groups = (
            jd_analysis.get(
                "keyword_groups",
                {},
            )
            or {}
        )

        if not keyword_groups:
            keyword_groups = {
                "technical": [],
                "domain": [],
                "concepts": [],
            }

        st.session_state.jd_text = jd_text
        st.session_state.jd_keywords = jd_keywords
        st.session_state.jd_keyword_groups = keyword_groups
        st.session_state.jd_keyword_retrieved = (
            rag_gemini_result.get(
                "retrieved",
                [],
            )
            if rag_gemini_result
            else []
        )
        st.session_state.jd_keyword_source = (
            "Local hybrid NLP"
            if not use_jd_rag
            else (
                "Robust section-aware NLP + RAG + Gemini"
                if rag_gemini_result.get("used_gemini")
                else "Section-aware NLP + RAG fallback"
            )
        )
        st.session_state.jd_keyword_warning = (
            rag_gemini_result.get(
                "warning",
                "",
            )
            if rag_gemini_result
            else ""
        )
        st.session_state.job_fingerprint = current_jd_fingerprint

        # Keep the deterministic analysis authoritative while exposing
        # the additive RAG + Gemini layer.
        if rag_gemini_result:
            rag_analysis = rag_gemini_result.get(
                "analysis",
                {},
            ) or {}

            # V4 analysis is section-aware and can replace the base V3
            # analysis object without changing the candidate matching API.
            for key in [
                "job_title",
                "title",
                "role",
                "required_skills",
                "preferred_skills",
                "experience",
                "sections",
                "section_presence",
                "required_evidence",
                "preferred_evidence",
                "responsibilities",
                "education",
                "summary",
            ]:
                if key in rag_analysis:
                    jd_analysis[key] = rag_analysis[key]

            jd_analysis["rag_retrieved"] = rag_gemini_result.get(
                "retrieved",
                [],
            )
            jd_analysis["rag_gemini"] = rag_gemini_result.get(
                "gemini",
                None,
            )

            st.session_state.jd_rag_gemini_warning = (
                rag_gemini_result.get(
                    "warning",
                    "",
                )
            )

        # ----------------------------------------------------
        # JD REQUIREMENT MATRIX V7
        # ----------------------------------------------------
        jd_matrix = build_requirement_matrix(
            jd_analysis
        )

        st.session_state.jd_requirement_matrix = jd_matrix

        st.session_state.jd_analysis = jd_analysis

        # ----------------------------------------------------
        # RESUME EXTRACTION
        # ----------------------------------------------------
        valid_resume_data = []
        extraction_progress = st.progress(0)

        total_resumes = len(resume_files)

        for index, resume in enumerate(resume_files):
            resume_name = getattr(
                resume,
                "name",
                f"Resume_{index + 1}",
            )

            try:
                resume_text = extract_resume_text(resume)

                if resume_text:
                    valid_resume_data.append(
                        (
                            resume_name,
                            resume_text,
                        )
                    )
                else:
                    st.warning(
                        f"No readable text found in {resume_name}."
                    )

            except Exception as error:
                st.error(
                    f"Error reading {resume_name}: {error}"
                )

            extraction_progress.progress(
                (index + 1) / total_resumes
            )

        extraction_progress.empty()

        if not valid_resume_data:
            st.error("No readable resumes were found.")
            st.stop()

        # ----------------------------------------------------
        # BATCH ATS / SEMANTIC ANALYSIS
        # ----------------------------------------------------
        with st.spinner(
            f"Running batch AI analysis for "
            f"{len(valid_resume_data)} candidate(s)..."
        ):
            batch_results = analyze_resumes_batch(
                jd_text,
                valid_resume_data,
                top_n=top_n_keywords,
            )

        if not batch_results:
            st.error(
                "The analysis engine returned no results."
            )
            st.stop()

        analysis_lookup = {
            result["resume_name"]: result
            for result in batch_results
        }

        results = []
        pdf_reports = {}

        # ----------------------------------------------------
        # CANDIDATE INTELLIGENCE
        # ----------------------------------------------------
        progress = st.progress(0)
        status_text = st.empty()

        total_candidates = len(valid_resume_data)

        for index, (resume_name, resume_text) in enumerate(
            valid_resume_data
        ):
            status_text.write(
                f"Building candidate intelligence: {resume_name}"
            )

            try:
                analysis_result = analysis_lookup.get(
                    resume_name
                )

                if not analysis_result:
                    continue

                # ATS
                ats_result = analyze_ats(
                    skill_score=analysis_result.get(
                        "skill_score",
                        0,
                    ),
                    semantic_score=analysis_result.get(
                        "semantic_score",
                        0,
                    ),
                    lexical_score=analysis_result.get(
                        "lexical_score",
                        0,
                    ),
                    matched_keywords=analysis_result.get(
                        "matched_keywords",
                        [],
                    ),
                    missing_keywords=analysis_result.get(
                        "missing_keywords",
                        [],
                    ),
                    matched_skills=analysis_result.get(
                        "matched_skills",
                        [],
                    ),
                    missing_skills=analysis_result.get(
                        "missing_skills",
                        [],
                    ),
                )

                # Intelligence
                intelligence = analyze_resume_intelligence(
                    resume_text,
                    source_filename=resume_name,
                )

                # Quality
                quality = analyze_resume_quality(
                    resume_text,
                    sections=intelligence.get(
                        "sections",
                        {},
                    ),
                )

                # Requirements
                requirements = analyze_candidate_requirements(
                    candidate_intelligence=intelligence,
                    jd_analysis=jd_analysis,
                )

                # Candidate name
                candidate_name = str(
                    intelligence.get("name", "")
                ).strip()

                if not candidate_name:
                    fallback = os.path.splitext(
                        os.path.basename(resume_name)
                    )[0]

                    fallback = re.sub(
                        r"[_-]+",
                        " ",
                        fallback,
                    )

                    fallback = re.sub(
                        r"\b(resume|cv)\b",
                        " ",
                        fallback,
                        flags=re.IGNORECASE,
                    )

                    fallback = re.sub(
                        r"\s+",
                        " ",
                        fallback,
                    ).strip()

                    candidate_name = (
                        fallback.title()
                        if fallback
                        else "Candidate"
                    )

                required_data = requirements.get(
                    "required_skills",
                    {},
                )

                preferred_data = requirements.get(
                    "preferred_skills",
                    {},
                )

                statistics = intelligence.get(
                    "statistics",
                    {},
                )

                # Cache full candidate object.
                st.session_state.resume_cache[
                    resume_name
                ] = {
                    "text": resume_text,
                    "analysis": analysis_result,
                    "ats": ats_result,
                    "intelligence": intelligence,
                    "quality": quality,
                    "requirements": requirements,
                }

                result_row = {
                    "Candidate": candidate_name,
                    "Resume": resume_name,
                    "ATS Score": ats_result.get(
                        "ats_score",
                        0,
                    ),
                    "Resume Quality": quality.get(
                        "quality_score",
                        0,
                    ),
                    "Requirement Match": requirements.get(
                        "requirement_match_score"
                    ),
                    "Semantic Score": ats_result.get(
                        "semantic_score",
                        0,
                    ),
                    "Lexical Score": ats_result.get(
                        "lexical_score",
                        0,
                    ),
                    "Skill Score": ats_result.get(
                        "skill_score",
                        0,
                    ),
                    "Keyword Score": ats_result.get(
                        "keyword_score",
                        0,
                    ),
                    "Experience Years": intelligence.get(
                        "experience_years",
                        0,
                    ),
                    "Skills Detected": len(
                        intelligence.get(
                            "all_skills",
                            [],
                        )
                    ),
                    "Required Skills Matched": len(
                        required_data.get(
                            "matched",
                            [],
                        )
                    ),
                    "Required Skills Missing": len(
                        required_data.get(
                            "missing",
                            [],
                        )
                    ),
                    "Preferred Skills Matched": len(
                        preferred_data.get(
                            "matched",
                            [],
                        )
                    ),
                    "Section Completeness": intelligence.get(
                        "section_completeness",
                        0,
                    ),
                    "Contact Completeness": intelligence.get(
                        "contact_completeness",
                        0,
                    ),
                    "Words": statistics.get(
                        "words",
                        0,
                    ),
                    "Email": intelligence.get(
                        "email",
                        "",
                    ),
                    "Phone": intelligence.get(
                        "phone",
                        "",
                    ),
                    "LinkedIn": intelligence.get(
                        "linkedin",
                        "",
                    ),
                    "GitHub": intelligence.get(
                        "github",
                        "",
                    ),
                    "Strengths": " | ".join(
                        ats_result.get(
                            "strengths",
                            [],
                        )
                    ),
                    "Suggestions": " | ".join(
                        ats_result.get(
                            "suggestions",
                            [],
                        )
                    ),
                }

                results.append(result_row)

                # Persist candidate metadata while preserving existing recruiter workflow.
                upsert_candidate(
                    job_fingerprint=st.session_state.job_fingerprint,
                    resume_name=resume_name,
                    candidate_name=candidate_name,
                    email=intelligence.get("email", ""),
                    phone=intelligence.get("phone", ""),
                    linkedin=intelligence.get("linkedin", ""),
                    github=intelligence.get("github", ""),
                    ats_score=ats_result.get("ats_score", 0),
                    requirement_score=requirements.get("requirement_match_score"),
                    quality_score=quality.get("quality_score", 0),
                    semantic_score=ats_result.get("semantic_score", 0),
                    skill_score=ats_result.get("skill_score", 0),
                    keyword_score=ats_result.get("keyword_score", 0),
                    experience_years=intelligence.get("experience_years", 0),
                )

                # PDF report
                pdf_buffer = generate_pdf_report(
                    candidate_name=candidate_name,
                    resume_name=resume_name,
                    ats=ats_result,
                    intelligence=intelligence,
                    quality=quality,
                    requirements=requirements,
                    analysis_result=analysis_result,
                )

                pdf_reports[
                    f"{safe_filename(resume_name)}_report.pdf"
                ] = pdf_buffer

            except Exception as error:
                st.error(
                    f"Error analyzing {resume_name}: {error}"
                )

            progress.progress(
                (index + 1) / total_candidates
            )

        status_text.empty()
        progress.empty()

        if not results:
            st.error(
                "No candidates could be analyzed successfully."
            )
            st.stop()

        df = pd.DataFrame(results)

        df = df.sort_values(
            "ATS Score",
            ascending=False,
        ).reset_index(drop=True)

        df.insert(
            0,
            "Rank",
            range(1, len(df) + 1),
        )

        st.session_state.analysis_results = df
        st.session_state.pdf_reports = pdf_reports

        st.session_state.candidate_workflow = initialize_workflow(
            df["Resume"].astype(str).tolist(),
            load_workflow(
                st.session_state.job_fingerprint,
                df["Resume"].astype(str).tolist(),
            ),
        )

        st.success(
            f"Successfully analyzed {len(df)} candidate(s)."
        )


# ============================================================
# DASHBOARD + CANDIDATE 360 + RECRUITER WORKFLOW
# ============================================================

if st.session_state.analysis_results is not None:
    df = st.session_state.analysis_results.copy()

    # Keep workflow synchronized with the persistent recruiter database.
    st.session_state.candidate_workflow = initialize_workflow(
        df["Resume"].astype(str).tolist(),
        sync_workflow_from_database(df),
    )

    workflow = st.session_state.candidate_workflow

    # --------------------------------------------------------
    # Add live workflow fields to dataframe
    # --------------------------------------------------------
    df["Status"] = df["Resume"].apply(
        lambda x: workflow.get(
            str(x),
            {"status": "New"},
        ).get("status", "New")
    )

    df["Shortlisted"] = df["Resume"].apply(
        lambda x: bool(
            workflow.get(
                str(x),
                {"shortlisted": False},
            ).get("shortlisted", False)
        )
    )

    df["Recruiter Notes"] = df["Resume"].apply(
        lambda x: workflow.get(
            str(x),
            {"notes": ""},
        ).get("notes", "")
    )

    st.divider()
    st.header("📊 Recruiter Intelligence Dashboard")

    dashboard_tab, analytics_tab, kanban_tab, profile_tab, brief_tab, copilot_tab, comparison_tab, interview_tab, communication_tab, workflow_tab = st.tabs(
        [
            "📊 Dashboard",
            "📈 Analytics",
            "📌 Pipeline",
            "👤 Candidate 360°",
            "🧠 Screening Brief",
            "🤖 Recruiter Copilot",
            "⚖️ Compare Candidates",
            "🗓️ Interviews",
            "📨 Communications",
            "🎯 Recruiter Workflow",
        ]
    )

    # ========================================================
    # DASHBOARD TAB
    # ========================================================
    with dashboard_tab:

        control1, control2 = st.columns(2)

        with control1:
            dashboard_search = st.text_input(
                "🔎 Search Candidate or Resume",
                placeholder="e.g. Adhyan or resume_01",
                key="dashboard_search",
            )

        with control2:
            sort_by = st.selectbox(
                "↕️ Sort Candidates By",
                [
                    "ATS Score",
                    "Resume Quality",
                    "Requirement Match",
                    "Skill Score",
                    "Semantic Score",
                    "Keyword Score",
                    "Experience Years",
                ],
                key="dashboard_sort_by",
            )

        dashboard_df = df.copy()

        if dashboard_search:
            query = dashboard_search.lower().strip()

            candidate_match = (
                dashboard_df["Candidate"].astype(str).str.lower().str.contains(
                    query,
                    na=False,
                )
            )

            resume_match = (
                dashboard_df["Resume"].astype(str).str.lower().str.contains(
                    query,
                    na=False,
                )
            )

            dashboard_df = dashboard_df[
                candidate_match | resume_match
            ]

        dashboard_df = dashboard_df[
            pd.to_numeric(
                dashboard_df["ATS Score"],
                errors="coerce",
            ).fillna(0) >= minimum_ats
        ]

        dashboard_df = dashboard_df.sort_values(
            sort_by,
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        dashboard_df["Rank"] = range(1, len(dashboard_df) + 1)

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------
        total_candidates = len(dashboard_df)

        average_ats = (
            pd.to_numeric(
                dashboard_df["ATS Score"],
                errors="coerce",
            ).mean()
            if total_candidates
            else 0
        )

        average_quality = (
            pd.to_numeric(
                dashboard_df["Resume Quality"],
                errors="coerce",
            ).mean()
            if total_candidates
            else 0
        )

        average_skill = (
            pd.to_numeric(
                dashboard_df["Skill Score"],
                errors="coerce",
            ).mean()
            if total_candidates
            else 0
        )

        shortlisted_count = int(
            dashboard_df["Shortlisted"].sum()
        ) if total_candidates else 0

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            st.metric("👥 Candidates", total_candidates)

        with kpi2:
            st.metric("🎯 Average ATS", f"{average_ats:.1f}%")

        with kpi3:
            st.metric("📄 Resume Quality", f"{average_quality:.1f}%")

        with kpi4:
            st.metric("🛠️ Skill Match", f"{average_skill:.1f}%")

        with kpi5:
            st.metric("⭐ Shortlisted", shortlisted_count)

        # ----------------------------------------------------
        # Ranking table
        # ----------------------------------------------------
        st.subheader("🏆 Candidate Ranking")

        ranking_columns = [
            "Rank",
            "Candidate",
            "Resume",
            "Status",
            "Shortlisted",
            "ATS Score",
            "Resume Quality",
            "Requirement Match",
            "Semantic Score",
            "Skill Score",
            "Keyword Score",
            "Experience Years",
            "Skills Detected",
            "Email",
            "Phone",
            "LinkedIn",
            "GitHub",
        ]

        ranking_columns = [
            column
            for column in ranking_columns
            if column in dashboard_df.columns
        ]

        ranking_df = dashboard_df[ranking_columns].copy()

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ATS Score": st.column_config.NumberColumn(
                    "ATS Score",
                    format="%.1f%%",
                ),
                "Resume Quality": st.column_config.NumberColumn(
                    "Resume Quality",
                    format="%.1f%%",
                ),
                "Requirement Match": st.column_config.NumberColumn(
                    "Requirement Match",
                    format="%.1f%%",
                ),
                "Semantic Score": st.column_config.NumberColumn(
                    "Semantic Score",
                    format="%.1f%%",
                ),
                "Skill Score": st.column_config.NumberColumn(
                    "Skill Score",
                    format="%.1f%%",
                ),
                "Keyword Score": st.column_config.NumberColumn(
                    "Keyword Score",
                    format="%.1f%%",
                ),
                "Experience Years": st.column_config.NumberColumn(
                    "Experience Years",
                    format="%.1f",
                ),
                "LinkedIn": st.column_config.LinkColumn(
                    "LinkedIn",
                    display_text="Open LinkedIn",
                ),
                "GitHub": st.column_config.LinkColumn(
                    "GitHub",
                    display_text="Open GitHub",
                ),
                "Shortlisted": st.column_config.CheckboxColumn(
                    "Shortlisted",
                ),
            },
        )

        if dashboard_df.empty:
            st.info("No candidates match the current filters.")
        else:
            # ------------------------------------------------
            # Charts
            # ------------------------------------------------
            st.subheader("📈 ATS Score Comparison")

            ats_chart = dashboard_df[["Candidate", "ATS Score"]].copy()
            ats_chart = ats_chart.set_index("Candidate")
            st.bar_chart(ats_chart)

            st.subheader("🧩 ATS Component Comparison")

            component_df = dashboard_df[
                [
                    "Candidate",
                    "Skill Score",
                    "Semantic Score",
                    "Lexical Score",
                    "Keyword Score",
                ]
            ].copy().set_index("Candidate")

            st.bar_chart(component_df)

            st.subheader("📄 Resume Quality Comparison")

            quality_df = dashboard_df[
                [
                    "Candidate",
                    "Resume Quality",
                    "Section Completeness",
                    "Contact Completeness",
                ]
            ].copy().set_index("Candidate")

            st.bar_chart(quality_df)

        # ----------------------------------------------------
        # JD intelligence
        # ----------------------------------------------------
        with st.expander(
            "🔑 Job Description Intelligence",
            expanded=False,
        ):
            jd_analysis = st.session_state.jd_analysis

            if not jd_analysis:
                st.info("JD intelligence is unavailable.")
            else:
                jd1, jd2 = st.columns(2)

                with jd1:
                    st.markdown("### Required Skills")
                    required = jd_analysis.get("required_skills", [])
                    st.write(", ".join(required) or "None detected.")

                    required_evidence = jd_analysis.get(
                        "required_evidence",
                        {},
                    ) or {}

                    if required_evidence:
                        with st.expander(
                            "🔎 Required-skill evidence",
                            expanded=False,
                        ):
                            for skill, evidence_lines in required_evidence.items():
                                st.write(f"**{skill}**")
                                for line in evidence_lines[:3]:
                                    st.caption(line)

                    st.markdown("### Experience Requirement")
                    experience = jd_analysis.get("experience", {})
                    st.write(
                        experience.get("raw_matches", [])
                        or "No explicit experience requirement detected."
                    )

                with jd2:
                    st.markdown("### Preferred Skills")
                    preferred = jd_analysis.get("preferred_skills", [])
                    st.write(", ".join(preferred) or "None detected.")

                    preferred_evidence = jd_analysis.get(
                        "preferred_evidence",
                        {},
                    ) or {}

                    if preferred_evidence:
                        with st.expander(
                            "🔎 Preferred-skill evidence",
                            expanded=False,
                        ):
                            for skill, evidence_lines in preferred_evidence.items():
                                st.write(f"**{skill}**")
                                for line in evidence_lines[:3]:
                                    st.caption(line)

                    st.markdown("### JD Keywords")

                    keyword_source = st.session_state.get(
                        "jd_keyword_source",
                        "Local hybrid NLP",
                    )
                    st.caption(
                        f"Extraction: **{keyword_source}**"
                    )

                    keyword_groups = st.session_state.get(
                        "jd_keyword_groups",
                        {},
                    ) or {}

                    technical = keyword_groups.get(
                        "technical",
                        [],
                    )
                    domain = keyword_groups.get(
                        "domain",
                        [],
                    )
                    concepts = keyword_groups.get(
                        "concepts",
                        [],
                    )

                    if technical:
                        st.markdown("**🛠️ Technical**")
                        st.write(", ".join(technical))

                    if domain:
                        st.markdown("**🌐 Domain**")
                        st.write(", ".join(domain))

                    if concepts:
                        st.markdown("**🧩 Concepts**")
                        st.write(", ".join(concepts))

                    # Fallback if a custom model/group result is empty.
                    if not any([
                        technical,
                        domain,
                        concepts,
                    ]):
                        st.write(
                            ", ".join(
                                st.session_state.jd_keywords
                            )
                            or "None detected."
                        )

                    warning = st.session_state.get(
                        "jd_keyword_warning",
                        "",
                    )

                    if warning:
                        st.info(warning)

                    # ------------------------------------------------
                    # Section-aware JD structure
                    # ------------------------------------------------
                    sections = jd_analysis.get(
                        "sections",
                        {},
                    ) or {}

                    if sections:
                        st.markdown("### 🧭 JD Structure Detected")

                        structure_labels = {
                            "summary": "Summary",
                            "required": "Required",
                            "preferred": "Preferred",
                            "skills": "Skills",
                            "experience": "Experience",
                            "education": "Education",
                            "responsibilities": "Responsibilities",
                            "general": "General",
                        }

                        section_names = [
                            structure_labels.get(
                                name,
                                name.title(),
                            )
                            for name in sections.keys()
                        ]

                        st.caption(
                            "Sections detected: "
                            + " • ".join(section_names)
                        )

                        expected_sections = {
                            "required",
                            "preferred",
                            "responsibilities",
                            "experience",
                            "education",
                        }

                        detected_core = expected_sections.intersection(
                            set(sections.keys())
                        )

                        if detected_core:
                            st.success(
                                f"Detected {len(detected_core)} core JD section(s) "
                                "from the extracted document text."
                            )

                            missing_core = sorted(
                                expected_sections
                                - detected_core
                            )

                            if missing_core:
                                st.caption(
                                    "Not explicitly recovered locally: "
                                    + ", ".join(
                                        structure_labels.get(
                                            item,
                                            item.title(),
                                        )
                                        for item in missing_core
                                    )
                                    + ". Gemini can still summarize "
                                      "these concepts from retrieved JD evidence."
                                )
                        else:
                            st.warning(
                                "The PDF text did not expose clear section boundaries. "
                                "Skill and Gemini analysis can still operate on the JD text."
                            )

                        with st.expander(
                            "📚 View JD section evidence",
                            expanded=False,
                        ):
                            for section_name, lines in sections.items():
                                display_name = structure_labels.get(
                                    section_name,
                                    section_name.title(),
                                )

                                st.markdown(
                                    f"**{display_name}**"
                                )

                                for line in lines[:8]:
                                    st.write(
                                        f"- {line}"
                                    )

                    # ------------------------------------------------
                    # RAG + Gemini JD Intelligence
                    # ------------------------------------------------
                    rag_result = st.session_state.get(
                        "jd_rag_gemini",
                        {},
                    ) or {}

                    gemini_jd = (
                        rag_result.get(
                            "gemini"
                        )
                        if isinstance(
                            rag_result,
                            dict,
                        )
                        else None
                    )

                    if use_jd_rag or gemini_jd:
                        st.markdown(
                            "### 🤖 AI JD Intelligence"
                        )

                        rag_warning = st.session_state.get(
                            "jd_rag_gemini_warning",
                            "",
                        )

                        if rag_warning:
                            st.info(rag_warning)

                        if gemini_jd:
                            detected_by_ai = []

                            if gemini_jd.get("responsibility_themes"):
                                detected_by_ai.append(
                                    "Responsibilities"
                                )

                            if gemini_jd.get("qualification_keywords"):
                                detected_by_ai.append(
                                    "Qualifications"
                                )

                            if gemini_jd.get("domain"):
                                detected_by_ai.append(
                                    "Domain"
                                )

                            if detected_by_ai:
                                st.caption(
                                    "AI context detected: "
                                    + " • ".join(
                                        detected_by_ai
                                    )
                                )

                            g1, g2 = st.columns(2)

                            with g1:
                                st.markdown("**Role Summary**")
                                st.write(
                                    gemini_jd.get(
                                        "role_summary",
                                        "Not detected",
                                    )
                                )

                                st.markdown("**Seniority**")
                                st.write(
                                    gemini_jd.get(
                                        "seniority",
                                        "Not explicitly detected",
                                    )
                                )

                                st.markdown("**Domain**")
                                st.write(
                                    ", ".join(
                                        gemini_jd.get(
                                            "domain",
                                            [],
                                        )
                                    )
                                    or "Not detected"
                                )

                            with g2:
                                st.markdown("**Responsibility Themes**")
                                st.write(
                                    ", ".join(
                                        gemini_jd.get(
                                            "responsibility_themes",
                                            [],
                                        )
                                    )
                                    or "Not detected"
                                )

                                st.markdown("**Soft Skills**")
                                st.write(
                                    ", ".join(
                                        gemini_jd.get(
                                            "soft_skills",
                                            [],
                                        )
                                    )
                                    or "Not detected"
                                )

                                st.markdown("**Qualification Keywords**")
                                st.write(
                                    ", ".join(
                                        gemini_jd.get(
                                            "qualification_keywords",
                                            [],
                                        )
                                    )
                                    or "Not detected"
                                )

                            with st.expander(
                                "🧠 Gemini-enriched keywords",
                                expanded=False,
                            ):
                                for label, key in [
                                    ("Technical", "technical_keywords"),
                                    ("Domain", "domain_keywords"),
                                    ("Concepts", "concept_keywords"),
                                ]:
                                    values = gemini_jd.get(
                                        key,
                                        [],
                                    ) or []

                                    if values:
                                        st.markdown(
                                            f"**{label}**"
                                        )
                                        st.write(
                                            ", ".join(values)
                                        )

                            with st.expander(
                                "📌 Gemini evidence notes",
                                expanded=False,
                            ):
                                notes = gemini_jd.get(
                                    "evidence_notes",
                                    [],
                                ) or []

                                if notes:
                                    for note in notes:
                                        st.write(
                                            f"- {note}"
                                        )
                                else:
                                    st.caption(
                                        "No evidence notes returned."
                                    )

                            st.caption(
                                "Grounding: Retrieved JD evidence → Gemini. "
                                "Deterministic required/preferred skills remain authoritative."
                            )

                    retrieved = (
                        jd_analysis.get(
                            "rag_retrieved",
                            [],
                        )
                        or st.session_state.get(
                            "jd_keyword_retrieved",
                            [],
                        )
                    )

                    # ------------------------------------------------
                    # JD REQUIREMENT MATRIX V7
                    # ------------------------------------------------
                    st.markdown(
                        "### 📋 JD Requirement Matrix"
                    )

                    jd_matrix = st.session_state.get(
                        "jd_requirement_matrix",
                        [],
                    ) or []

                    if jd_matrix:
                        matrix_summary = build_requirement_summary(
                            jd_matrix
                        )

                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)

                        with mc1:
                            st.metric(
                                "Must Have",
                                matrix_summary.get(
                                    "required",
                                    0,
                                ),
                            )

                        with mc2:
                            st.metric(
                                "Preferred",
                                matrix_summary.get(
                                    "preferred",
                                    0,
                                ),
                            )

                        with mc3:
                            st.metric(
                                "Experience",
                                matrix_summary.get(
                                    "experience",
                                    0,
                                ),
                            )

                        with mc4:
                            st.metric(
                                "Education",
                                matrix_summary.get(
                                    "education",
                                    0,
                                ),
                            )

                        with mc5:
                            st.metric(
                                "Context",
                                (
                                    matrix_summary.get(
                                        "responsibilities",
                                        0,
                                    )
                                    + matrix_summary.get(
                                        "domain",
                                        0,
                                    )
                                    + matrix_summary.get(
                                        "concepts",
                                        0,
                                    )
                                    + matrix_summary.get(
                                        "soft_skills",
                                        0,
                                    )
                                ),
                            )

                        matrix_df = requirements_to_dataframe(
                            jd_matrix
                        )

                        st.dataframe(
                            matrix_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Requirement": st.column_config.TextColumn(
                                    "Requirement",
                                    width="large",
                                ),
                                "Category": st.column_config.TextColumn(
                                    "Category",
                                ),
                                "Priority": st.column_config.TextColumn(
                                    "Priority",
                                ),
                                "Source": st.column_config.TextColumn(
                                    "Source",
                                ),
                                "Section": st.column_config.TextColumn(
                                    "Section",
                                ),
                                "Evidence": st.column_config.TextColumn(
                                    "Evidence",
                                    width="large",
                                ),
                            },
                        )

                        with st.expander(
                            "📄 Export JD Requirement Matrix",
                            expanded=False,
                        ):
                            st.download_button(
                                "Download Requirement Matrix",
                                data=requirement_matrix_to_text(
                                    jd_matrix
                                ),
                                file_name="jd_requirement_matrix.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )
                    else:
                        st.info(
                            "No structured JD requirements were extracted."
                        )

                    if retrieved:
                        with st.expander(
                            "🔎 Retrieved JD Evidence",
                            expanded=False,
                        ):
                            st.caption(
                                "These JD snippets were retrieved before "
                                "optional Gemini keyword enrichment."
                            )

                            for index, item in enumerate(
                                retrieved,
                                start=1,
                            ):
                                st.write(
                                    f"**{index}.** "
                                    f"{item.get('text', '')}"
                                )
                                st.caption(
                                    "Retrieval score: "
                                    f"{float(item.get('retrieval_score', 0)):.3f}"
                                )

    # ========================================================
    # RECRUITER ANALYTICS TAB
    # ========================================================
    with analytics_tab:
        st.subheader("📈 Recruiter Analytics")
        st.caption(
            "Descriptive analytics across pipeline status, matching signals, "
            "required-skill gaps, requirement coverage, interviews and communications."
        )

        current_interviews = (
            list_interviews(
                st.session_state.job_fingerprint
            )
            if st.session_state.get(
                "job_fingerprint",
                "",
            )
            else []
        )

        current_communications = (
            list_communications(
                st.session_state.job_fingerprint
            )
            if st.session_state.get(
                "job_fingerprint",
                "",
            )
            else []
        )

        analytics_snapshot = build_analytics_snapshot(
            df=df,
            resume_cache=st.session_state.get(
                "resume_cache",
                {},
            ),
            interviews=current_interviews,
            communications=current_communications,
        )

        shortlist = analytics_snapshot["shortlisted"]
        interview_summary = analytics_snapshot["interviews"]

        a1, a2, a3, a4, a5 = st.columns(5)

        with a1:
            st.metric(
                "👥 Candidates",
                analytics_snapshot["candidate_count"],
            )

        with a2:
            st.metric(
                "⭐ Shortlisted",
                shortlist["shortlisted"],
            )

        with a3:
            st.metric(
                "📌 Shortlist Rate",
                f"{shortlist['shortlist_rate']:.1f}%",
            )

        with a4:
            st.metric(
                "🗓️ Interviews",
                interview_summary["total"],
            )

        with a5:
            st.metric(
                "✅ Completed",
                interview_summary["completed"],
            )

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("### 📌 Pipeline Distribution")

            pipeline_df = analytics_snapshot["pipeline"]

            st.dataframe(
                pipeline_df,
                use_container_width=True,
                hide_index=True,
            )

            if not pipeline_df.empty:
                st.bar_chart(
                    pipeline_df.set_index("Status")
                )

        with right:
            st.markdown("### 📊 Matching Signal Summary")

            score_df = analytics_snapshot["scores"]

            if score_df.empty:
                st.info(
                    "No matching-signal statistics are available."
                )
            else:
                st.dataframe(
                    score_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    score_df[
                        [
                            "Metric",
                            "Average",
                        ]
                    ].set_index("Metric")
                )

        st.divider()

        st.markdown("### 🔍 Required Skill Gap Trends")

        gap_df = analytics_snapshot[
            "required_skill_gaps"
        ]

        if gap_df.empty:
            st.success(
                "No missing required-skill signals were found."
            )
        else:
            gap_left, gap_right = st.columns(2)

            with gap_left:
                st.dataframe(
                    gap_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with gap_right:
                st.bar_chart(
                    gap_df.set_index(
                        "Required Skill"
                    )
                )

        preferred_gap_df = analytics_snapshot[
            "preferred_skill_gaps"
        ]

        if not preferred_gap_df.empty:
            with st.expander(
                "⭐ Preferred Skill Gap Trends",
                expanded=False,
            ):
                st.dataframe(
                    preferred_gap_df,
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        st.markdown(
            "### 🧩 Required Skill Coverage Across Candidates"
        )

        coverage_df = analytics_snapshot[
            "required_skill_coverage"
        ]

        if coverage_df.empty:
            st.info(
                "No required-skill coverage data is available."
            )
        else:
            st.dataframe(
                coverage_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Coverage is derived from the existing requirement-matching "
                "results; no additional candidate score is created."
            )

        st.divider()

        interview_col, communication_col = st.columns(2)

        with interview_col:
            st.markdown("### 🗓️ Interview Activity")

            interview_table = pd.DataFrame(
                [
                    {
                        "Status": "Scheduled",
                        "Count": interview_summary["scheduled"],
                    },
                    {
                        "Status": "Completed",
                        "Count": interview_summary["completed"],
                    },
                    {
                        "Status": "Cancelled",
                        "Count": interview_summary["cancelled"],
                    },
                    {
                        "Status": "Rescheduled",
                        "Count": interview_summary["rescheduled"],
                    },
                    {
                        "Status": "Pending Feedback",
                        "Count": interview_summary["pending_feedback"],
                    },
                ]
            )

            st.dataframe(
                interview_table,
                use_container_width=True,
                hide_index=True,
            )

        with communication_col:
            st.markdown("### 📨 Communication Activity")

            communication_df = analytics_snapshot[
                "communications"
            ]

            if communication_df.empty:
                st.info(
                    "No recruiter communication history is available."
                )
            else:
                st.dataframe(
                    communication_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # ========================================================
    # KANBAN PIPELINE TAB
    # ========================================================
    with kanban_tab:
        st.subheader("📌 Recruiter Pipeline")
        st.caption(
            "Track candidates across the hiring pipeline. "
            "Use the arrow controls on each card to move a candidate "
            "between stages; changes are stored permanently."
        )

        pipeline_summary = board_summary(
            df.to_dict("records"),
            workflow,
        )

        p1, p2, p3, p4, p5, p6 = st.columns(6)

        with p1:
            st.metric("Total", pipeline_summary["total"])
        with p2:
            st.metric("New", pipeline_summary["new"])
        with p3:
            st.metric("Reviewing", pipeline_summary["reviewing"])
        with p4:
            st.metric("Shortlisted", pipeline_summary["shortlisted"])
        with p5:
            st.metric("Hold", pipeline_summary["hold"])
        with p6:
            st.metric("Rejected", pipeline_summary["rejected"])

        st.divider()

        kan1, kan2, kan3, kan4 = st.columns(4)

        with kan1:
            kanban_search = st.text_input(
                "🔎 Search",
                placeholder="Name or resume...",
                key="kanban_search",
            )

        with kan2:
            kanban_min_ats = st.slider(
                "Minimum ATS",
                0,
                100,
                0,
                key="kanban_min_ats",
            )

        with kan3:
            kanban_min_requirement = st.slider(
                "Minimum Requirement Match",
                0,
                100,
                0,
                key="kanban_min_requirement",
            )

        with kan4:
            kanban_shortlisted_only = st.checkbox(
                "⭐ Shortlisted only",
                key="kanban_shortlisted_only",
            )

        kan_sort = st.selectbox(
            "Sort cards by",
            [
                "ATS Score",
                "Requirement Match",
                "Resume Quality",
                "Experience Years",
                "Candidate",
            ],
            key="kanban_sort",
        )

        board_candidates = prepare_board_candidates(
            candidates=df.to_dict("records"),
            workflow=workflow,
            search_text=kanban_search,
            min_ats=kanban_min_ats,
            min_requirement=kanban_min_requirement,
            shortlisted_only=kanban_shortlisted_only,
            sort_by=kan_sort,
        )

        columns = split_into_columns(board_candidates)

        st.write(
            f"Showing **{len(board_candidates)}** of **{len(df)}** candidates"
        )

        # Inject lightweight CSS for the pipeline cards.
        st.markdown(
            """
            <style>
            .kanban-card {
                border: 1px solid rgba(128,128,128,.35);
                border-radius: 12px;
                padding: 14px;
                margin-bottom: 12px;
                background: rgba(128,128,128,.04);
            }
            .kanban-name {
                font-weight: 700;
                font-size: 1rem;
                margin-bottom: 3px;
            }
            .kanban-resume {
                font-size: .78rem;
                opacity: .70;
                margin-bottom: 10px;
                word-break: break-word;
            }
            .kanban-metric {
                font-size: .82rem;
                margin: 2px 0;
            }
            .kanban-note {
                font-size: .78rem;
                opacity: .78;
                margin-top: 8px;
                padding: 8px;
                border-radius: 8px;
                background: rgba(128,128,128,.08);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        board_cols = st.columns(len(KANBAN_STATUSES))

        for board_col, status_name in zip(board_cols, KANBAN_STATUSES):
            with board_col:
                stage_candidates = columns.get(status_name, [])

                st.markdown(
                    f"### {status_name} \n                    ({len(stage_candidates)})"
                )
                st.caption(STATUS_DESCRIPTIONS.get(status_name, ""))

                if not stage_candidates:
                    st.info("No candidates")
                    continue

                for card_index, candidate in enumerate(stage_candidates):
                    resume_name = str(candidate.get("Resume", ""))
                    candidate_name = str(candidate.get("Candidate", "Candidate"))
                    ats_score = float(candidate.get("ATS Score", 0) or 0)
                    requirement_score = candidate.get("Requirement Match")
                    quality_score = float(candidate.get("Resume Quality", 0) or 0)
                    experience = float(candidate.get("Experience Years", 0) or 0)
                    shortlisted = bool(candidate.get("workflow_shortlisted", False))
                    notes = str(candidate.get("workflow_notes", ""))

                    st.markdown(
                        f"""
                        <div class="kanban-card">
                            <div class="kanban-name">{candidate_name}</div>
                            <div class="kanban-resume">{resume_name}</div>
                            <div class="kanban-metric"><b>ATS:</b> {ats_score:.1f}%</div>
                            <div class="kanban-metric"><b>Requirement:</b> {float(requirement_score or 0):.1f}%</div>
                            <div class="kanban-metric"><b>Quality:</b> {quality_score:.1f}%</div>
                            <div class="kanban-metric"><b>Experience:</b> {experience:.1f} yrs</div>
                            <div class="kanban-metric"><b>Shortlisted:</b> {"Yes" if shortlisted else "No"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    safe_key = safe_filename(resume_name)

                    act1, act2, act3 = st.columns(3)

                    with act1:
                        prev_status = move_status(status_name, -1)
                        if status_name != prev_status:
                            if st.button(
                                "←",
                                key=f"kanban_left_{safe_key}_{card_index}",
                                help=f"Move to {prev_status}",
                                use_container_width=True,
                            ):
                                set_candidate_status(
                                    workflow,
                                    resume_name,
                                    prev_status,
                                )
                                persist_workflow_state(
                                    resume_name,
                                    workflow[resume_name],
                                )
                                st.session_state.candidate_workflow = workflow
                                st.rerun()
                        else:
                            st.button(
                                "·",
                                key=f"kanban_left_disabled_{safe_key}_{card_index}",
                                disabled=True,
                                use_container_width=True,
                            )

                    with act2:
                        if st.button(
                            "👤",
                            key=f"kanban_profile_{safe_key}_{card_index}",
                            help="Select this candidate in Candidate 360°",
                            use_container_width=True,
                        ):
                            st.session_state.selected_candidate_resume = resume_name
                            st.session_state.kanban_selected_candidate = resume_name
                            st.rerun()

                    with act3:
                        next_status = move_status(status_name, 1)
                        if status_name != next_status:
                            if st.button(
                                "→",
                                key=f"kanban_right_{safe_key}_{card_index}",
                                help=f"Move to {next_status}",
                                use_container_width=True,
                            ):
                                set_candidate_status(
                                    workflow,
                                    resume_name,
                                    next_status,
                                )
                                persist_workflow_state(
                                    resume_name,
                                    workflow[resume_name],
                                )
                                st.session_state.candidate_workflow = workflow
                                st.rerun()
                        else:
                            st.button(
                                "·",
                                key=f"kanban_right_disabled_{safe_key}_{card_index}",
                                disabled=True,
                                use_container_width=True,
                            )

                    star_label = "⭐ Unshortlist" if shortlisted else "⭐ Shortlist"
                    if st.button(
                        star_label,
                        key=f"kanban_star_{safe_key}_{card_index}",
                        use_container_width=True,
                    ):
                        set_shortlist(
                            workflow,
                            resume_name,
                            not shortlisted,
                        )
                        persist_workflow_state(
                            resume_name,
                            workflow[resume_name],
                        )
                        st.session_state.candidate_workflow = workflow
                        st.rerun()

                    if notes:
                        preview = notes.replace("\n", " ").strip()
                        if len(preview) > 90:
                            preview = preview[:87] + "..."
                        st.markdown(
                            f'<div class="kanban-note">📝 {preview}</div>',
                            unsafe_allow_html=True,
                        )

        selected_kanban_candidate = st.session_state.get(
            "kanban_selected_candidate",
            "",
        )

        if selected_kanban_candidate:
            selected_candidate_row = df[
                df["Resume"].astype(str) == str(selected_kanban_candidate)
            ]

            if not selected_candidate_row.empty:
                selected_candidate = selected_candidate_row.iloc[0]
                selected_resume_name = str(selected_candidate["Resume"])
                selected_state = workflow.get(
                    selected_resume_name,
                    {"status": "New", "shortlisted": False, "notes": ""},
                )

                st.divider()
                st.markdown("### 👤 Quick Candidate View")
                qc1, qc2, qc3, qc4 = st.columns(4)

                with qc1:
                    st.metric(
                        "Candidate",
                        str(selected_candidate.get("Candidate", "Candidate")),
                    )
                with qc2:
                    st.metric(
                        "ATS",
                        f"{float(selected_candidate.get('ATS Score', 0) or 0):.1f}%",
                    )
                with qc3:
                    st.metric(
                        "Requirement",
                        f"{float(selected_candidate.get('Requirement Match', 0) or 0):.1f}%",
                    )
                with qc4:
                    st.metric(
                        "Status",
                        selected_state.get("status", "New"),
                    )

                st.info(
                    "Candidate selected. Open the Candidate 360° tab to view the full profile."
                )

                if st.button(
                    "Clear Quick View",
                    key="clear_kanban_selection",
                ):
                    st.session_state.kanban_selected_candidate = ""
                    st.rerun()

    # ========================================================
    # CANDIDATE 360 TAB
    # ========================================================
    with profile_tab:
        st.subheader("👤 Candidate 360° View")

        profile_source = dashboard_df.copy()

        if profile_source.empty:
            profile_source = df.copy()

        profile_options = profile_source["Resume"].astype(str).tolist()

        if not profile_options:
            st.info("No candidate profiles are available.")
        else:
            current_selected = st.session_state.get(
                "selected_candidate_resume",
                "",
            )

            if current_selected not in profile_options:
                current_selected = profile_options[0]

            selected_resume = st.selectbox(
                "Select Candidate",
                profile_options,
                index=profile_options.index(current_selected),
                key="candidate_360_selector",
            )

            st.session_state.selected_candidate_resume = selected_resume

            selected_row = df[
                df["Resume"].astype(str) == str(selected_resume)
            ]

            if selected_row.empty:
                st.error("Candidate record not found.")
            else:
                row = selected_row.iloc[0].to_dict()
                cached = st.session_state.resume_cache.get(selected_resume)

                if not cached:
                    st.warning("Candidate details are unavailable.")
                else:
                    analysis_result = cached.get("analysis", {})
                    ats_result = cached.get("ats", {})
                    intelligence = cached.get("intelligence", {})
                    quality = cached.get("quality", {})
                    requirements = cached.get("requirements", {})

                    candidate_name = row.get(
                        "Candidate",
                        "Candidate",
                    )

                    state = workflow.get(
                        selected_resume,
                        {
                            "status": "New",
                            "shortlisted": False,
                            "notes": "",
                        },
                    )

                    snapshot = build_candidate_snapshot(
                        row,
                        intelligence,
                        ats_result,
                        quality,
                        requirements,
                    )

                    # ----------------------------------------
                    # Header + recruiter actions
                    # ----------------------------------------
                    header_left, header_right = st.columns([3, 1])

                    with header_left:
                        st.markdown(f"## {candidate_name}")
                        st.caption(selected_resume)

                    with header_right:
                        if state.get("shortlisted", False):
                            st.success("⭐ Shortlisted")
                        else:
                            st.info(
                                f"Status: {state.get('status', 'New')}"
                            )

                    action1, action2, action3 = st.columns(3)

                    with action1:
                        status_value = st.selectbox(
                            "Candidate Status",
                            STATUS_OPTIONS,
                            index=(
                                STATUS_OPTIONS.index(
                                    state.get("status", "New")
                                )
                                if state.get("status", "New") in STATUS_OPTIONS
                                else 0
                            ),
                            key=f"profile_status_{safe_filename(selected_resume)}",
                        )

                    with action2:
                        if state.get("shortlisted", False):
                            button_text = "⭐ Remove Shortlist"
                        else:
                            button_text = "⭐ Add to Shortlist"

                        if st.button(
                            button_text,
                            key=f"profile_shortlist_{safe_filename(selected_resume)}",
                            use_container_width=True,
                        ):
                            set_shortlist(
                                workflow,
                                selected_resume,
                                not state.get("shortlisted", False),
                            )
                            persist_workflow_state(
                                selected_resume,
                                workflow[selected_resume],
                            )
                            st.session_state.candidate_workflow = workflow
                            st.rerun()

                    with action3:
                        pdf_key = f"{safe_filename(selected_resume)}_report.pdf"
                        pdf_buffer = st.session_state.pdf_reports.get(pdf_key)

                        if pdf_buffer:
                            st.download_button(
                                "📥 Candidate PDF",
                                data=pdf_buffer.getvalue(),
                                file_name=pdf_key,
                                mime="application/pdf",
                                key=f"profile_pdf_{safe_filename(selected_resume)}",
                                use_container_width=True,
                            )

                    if status_value != state.get("status", "New"):
                        set_candidate_status(
                            workflow,
                            selected_resume,
                            status_value,
                        )
                        persist_workflow_state(
                            selected_resume,
                            workflow[selected_resume],
                        )
                        st.session_state.candidate_workflow = workflow
                        st.rerun()

                    # ----------------------------------------
                    # Contact
                    # ----------------------------------------
                    st.divider()
                    st.markdown("### 📇 Contact & Profile")

                    contact1, contact2, contact3, contact4 = st.columns(4)

                    with contact1:
                        st.markdown("**📧 Email**")
                        email = intelligence.get("email", "")
                        st.write(email or "Not detected")

                    with contact2:
                        st.markdown("**📱 Phone**")
                        phone = intelligence.get("phone", "")
                        st.write(phone or "Not detected")

                    with contact3:
                        st.markdown("**🔗 LinkedIn**")
                        linkedin = intelligence.get("linkedin", "")
                        if linkedin:
                            st.link_button("Open LinkedIn", linkedin)
                        else:
                            st.write("Not detected")

                    with contact4:
                        st.markdown("**💻 GitHub**")
                        github = intelligence.get("github", "")
                        if github:
                            st.link_button("Open GitHub", github)
                        else:
                            st.write("Not detected")

                    # ----------------------------------------
                    # Score cards
                    # ----------------------------------------
                    m1, m2, m3, m4, m5 = st.columns(5)

                    with m1:
                        st.metric(
                            "ATS",
                            f"{snapshot['ats_score']:.1f}%",
                        )

                    with m2:
                        st.metric(
                            "Requirement",
                            (
                                f"{safe_float(snapshot['requirement_score']):.1f}%"
                                if snapshot["requirement_score"] is not None
                                else "N/A"
                            ),
                        )

                    with m3:
                        st.metric(
                            "Semantic",
                            f"{snapshot['semantic_score']:.1f}%",
                        )

                    with m4:
                        st.metric(
                            "Skill Match",
                            f"{snapshot['skill_score']:.1f}%",
                        )

                    with m5:
                        st.metric(
                            "Resume Quality",
                            f"{snapshot['quality_score']:.1f}%",
                        )

                    # ----------------------------------------
                    # Screening summary
                    # ----------------------------------------
                    st.info(
                        "🧠 Screening Summary\n\n"
                        + build_screening_summary(snapshot)
                    )

                    profile_tabs = st.tabs(
                        [
                            "📋 Summary",
                            "🎯 Requirements",
                            "🛠️ Skills",
                            "🧠 Semantic",
                            "📄 Quality",
                            "📝 Recruiter Notes",
                            "🕒 Pipeline History",
                            "🔍 Explainability",
                        ]
                    )

                    # ========================================
                    # SUMMARY
                    # ========================================
                    with profile_tabs[0]:
                        sum1, sum2 = st.columns(2)

                        with sum1:
                            st.markdown("### 🎓 Education")
                            st.write(
                                intelligence.get("education", "")
                                or "Not detected"
                            )

                            st.markdown("### 💼 Experience")
                            experience_info = requirements.get(
                                "experience",
                                {},
                            )
                            st.write(
                                f"**Candidate experience:** "
                                f"{safe_float(intelligence.get('experience_years', 0)):.1f} years"
                            )
                            st.write(
                                f"**Requirement status:** "
                                f"{experience_info.get('status', 'Not specified')}"
                            )

                        with sum2:
                            st.markdown("### 🚀 Projects")
                            st.text(
                                intelligence.get("projects", "")
                                or "No Projects section detected."
                            )

                            st.markdown("### 🏆 Certifications")
                            st.text(
                                intelligence.get("certifications", "")
                                or "No Certifications section detected."
                            )

                        st.markdown("### 💡 ATS Strengths")
                        strengths = ats_result.get("strengths", [])
                        for strength in strengths:
                            st.success(strength)

                        st.markdown("### 🎯 ATS Suggestions")
                        suggestions = ats_result.get("suggestions", [])
                        for suggestion in suggestions:
                            st.info(suggestion)

                    # ========================================
                    # REQUIREMENTS
                    # ========================================
                    with profile_tabs[1]:
                        requirement_score = requirements.get(
                            "requirement_match_score"
                        )

                        if requirement_score is not None:
                            st.metric(
                                "Requirement Match",
                                f"{safe_float(requirement_score):.1f}%",
                            )

                        required_data = requirements.get(
                            "required_skills",
                            {},
                        )
                        preferred_data = requirements.get(
                            "preferred_skills",
                            {},
                        )

                        r1, r2 = st.columns(2)

                        with r1:
                            st.markdown("### ✅ Required Skills")
                            st.write(
                                "**Matched:** "
                                + (", ".join(required_data.get("matched", [])) or "None")
                            )
                            st.write(
                                "**Missing:** "
                                + (", ".join(required_data.get("missing", [])) or "None")
                            )
                            score = required_data.get("score")
                            if score is not None:
                                st.progress(
                                    max(0.0, min(1.0, safe_float(score) / 100.0))
                                )
                                st.caption(f"Coverage: {safe_float(score):.1f}%")

                        with r2:
                            st.markdown("### ⭐ Preferred Skills")
                            st.write(
                                "**Matched:** "
                                + (", ".join(preferred_data.get("matched", [])) or "None")
                            )
                            st.write(
                                "**Missing:** "
                                + (", ".join(preferred_data.get("missing", [])) or "None")
                            )
                            score = preferred_data.get("score")
                            if score is not None:
                                st.progress(
                                    max(0.0, min(1.0, safe_float(score) / 100.0))
                                )
                                st.caption(f"Coverage: {safe_float(score):.1f}%")

                        st.markdown("### 🎓 Education Requirement")
                        st.write(
                            requirements.get("education", {}).get(
                                "status",
                                "Not specified",
                            )
                        )

                        # ------------------------------------------------
                        # Candidate Requirement Coverage Matrix
                        # ------------------------------------------------
                        st.divider()
                        st.markdown(
                            "### 📋 Requirement Coverage"
                        )

                        candidate_matrix = (
                            build_candidate_requirement_matrix(
                                jd_analysis=st.session_state.get(
                                    "jd_analysis",
                                    {},
                                ),
                                candidate_requirements=requirements,
                                candidate_intelligence=intelligence,
                            )
                        )

                        if candidate_matrix:
                            candidate_matrix_df = requirements_to_dataframe(
                                candidate_matrix
                            )

                            candidate_matrix_df[
                                "Candidate Status"
                            ] = [
                                row.get(
                                    "candidate_status",
                                    "Context",
                                )
                                for row in candidate_matrix
                            ]

                            candidate_matrix_df[
                                "Match Source"
                            ] = [
                                row.get(
                                    "candidate_match_source",
                                    "",
                                )
                                for row in candidate_matrix
                            ]

                            # Put status immediately after priority.
                            preferred_columns = [
                                "Requirement",
                                "Category",
                                "Priority",
                                "Candidate Status",
                                "Match Source",
                                "Source",
                                "Section",
                                "Evidence",
                            ]

                            preferred_columns = [
                                col
                                for col in preferred_columns
                                if col in candidate_matrix_df.columns
                            ]

                            candidate_matrix_df = candidate_matrix_df[
                                preferred_columns
                            ]

                            st.dataframe(
                                candidate_matrix_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Requirement": st.column_config.TextColumn(
                                        "Requirement",
                                        width="large",
                                    ),
                                    "Candidate Status": st.column_config.TextColumn(
                                        "Candidate Status",
                                    ),
                                    "Match Source": st.column_config.TextColumn(
                                        "Match Source",
                                    ),
                                    "Evidence": st.column_config.TextColumn(
                                        "Evidence",
                                        width="large",
                                    ),
                                },
                            )

                            coverage_rows = [
                                row
                                for row in candidate_matrix
                                if row.get("category")
                                in {
                                    "Required Skill",
                                    "Preferred Skill",
                                    "Experience",
                                    "Education",
                                }
                            ]

                            matched_count = sum(
                                1
                                for row in coverage_rows
                                if row.get("candidate_status")
                                in {
                                    "Matched",
                                    "Detected in Resume",
                                    "Meets requirement",
                                    "Detected",
                                }
                            )

                            missing_count = sum(
                                1
                                for row in coverage_rows
                                if row.get("candidate_status")
                                in {
                                    "Missing",
                                    "Not detected",
                                }
                            )

                            cv1, cv2, cv3 = st.columns(3)

                            with cv1:
                                st.metric(
                                    "Covered",
                                    matched_count,
                                )

                            with cv2:
                                st.metric(
                                    "Needs Verification",
                                    missing_count,
                                )

                            with cv3:
                                st.metric(
                                    "Requirement Signals",
                                    len(coverage_rows),
                                )

                            with st.expander(
                                "📄 Export Candidate Requirement Coverage",
                                expanded=False,
                            ):
                                st.download_button(
                                    "Download Candidate Coverage",
                                    data=requirement_matrix_to_text(
                                        candidate_matrix
                                    ),
                                    file_name=(
                                        f"{safe_filename(selected_resume)}_"
                                        "requirement_coverage.txt"
                                    ),
                                    mime="text/plain",
                                    use_container_width=True,
                                )
                        else:
                            st.info(
                                "No requirement matrix is available for this candidate."
                            )

                    # ========================================
                    # SKILLS
                    # ========================================
                    with profile_tabs[2]:
                        detected_skills = intelligence.get("skills", {})

                        if detected_skills:
                            for category, skills in detected_skills.items():
                                st.markdown(
                                    f"**{category.replace('_', ' ').title()}**"
                                )
                                st.write(join_values(skills, "None"))
                        else:
                            st.info("No technical skills detected.")

                        st.divider()

                        sk1, sk2 = st.columns(2)

                        with sk1:
                            st.markdown("### ✅ Matched Skills")
                            st.write(
                                ", ".join(
                                    analysis_result.get(
                                        "matched_skills",
                                        [],
                                    )
                                ) or "None"
                            )

                        with sk2:
                            st.markdown("### ❌ Missing Skills")
                            st.write(
                                ", ".join(
                                    analysis_result.get(
                                        "missing_skills",
                                        [],
                                    )
                                ) or "None"
                            )

                        category_rows = []

                        for category, data in analysis_result.get(
                            "category_analysis",
                            {},
                        ).items():
                            category_rows.append(
                                {
                                    "Category": category.replace(
                                        "_",
                                        " ",
                                    ).title(),
                                    "Match (%)": data.get("score", 0),
                                    "Matched": ", ".join(
                                        data.get("matched", [])
                                    ) or "None",
                                    "Missing": ", ".join(
                                        data.get("missing", [])
                                    ) or "None",
                                }
                            )

                        if category_rows:
                            st.dataframe(
                                pd.DataFrame(category_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                    # ========================================
                    # SEMANTIC
                    # ========================================
                    with profile_tabs[3]:
                        st.metric(
                            "Semantic Match",
                            f"{safe_float(analysis_result.get('semantic_score', 0)):.1f}%",
                        )

                        section_scores = analysis_result.get(
                            "section_scores",
                            {},
                        )

                        if section_scores:
                            semantic_df = pd.DataFrame(
                                [
                                    {
                                        "Section": section.replace(
                                            "_",
                                            " ",
                                        ).title(),
                                        "Match (%)": score,
                                    }
                                    for section, score in section_scores.items()
                                ]
                            )

                            st.dataframe(
                                semantic_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.bar_chart(
                                semantic_df.set_index("Section")
                            )
                        else:
                            st.info(
                                "No section-level semantic comparison available."
                            )

                        k1, k2 = st.columns(2)

                        with k1:
                            st.markdown("### ✅ Matched Keywords")
                            st.write(
                                ", ".join(
                                    analysis_result.get(
                                        "matched_keywords",
                                        [],
                                    )
                                ) or "None"
                            )

                        with k2:
                            st.markdown("### ❌ Missing Keywords")
                            st.write(
                                ", ".join(
                                    analysis_result.get(
                                        "missing_keywords",
                                        [],
                                    )
                                ) or "None"
                            )

                    # ========================================
                    # QUALITY
                    # ========================================
                    with profile_tabs[4]:
                        quality_score = safe_float(
                            quality.get("quality_score", 0)
                        )

                        st.metric(
                            "Resume Quality",
                            f"{quality_score:.1f}%",
                        )

                        q1, q2 = st.columns(2)

                        with q1:
                            st.write(
                                f"**Section Completeness:** "
                                f"{safe_float(intelligence.get('section_completeness', 0)):.1f}%"
                            )
                            st.write(
                                f"**Contact Completeness:** "
                                f"{safe_float(intelligence.get('contact_completeness', 0)):.1f}%"
                            )
                            st.write(
                                f"**Word Count:** "
                                f"{quality.get('length', {}).get('word_count', 0)}"
                            )

                        with q2:
                            st.write(
                                f"**Action Verb Score:** "
                                f"{safe_float(quality.get('action_verbs', {}).get('score', 0)):.1f}%"
                            )
                            st.write(
                                f"**Quantification Score:** "
                                f"{safe_float(quality.get('quantification', {}).get('score', 0)):.1f}%"
                            )
                            st.write(
                                f"**Repetition Score:** "
                                f"{safe_float(quality.get('repetition', {}).get('score', 0)):.1f}%"
                            )

                        st.markdown("### Recommendations")
                        for recommendation in quality.get(
                            "recommendations",
                            [],
                        ):
                            st.info(recommendation)

                        issues = quality.get(
                            "extraction_issues",
                            [],
                        )

                        if issues:
                            st.markdown("### ⚠️ Extraction Observations")
                            for issue in issues:
                                st.warning(issue)

                    # ========================================
                    # RECRUITER NOTES
                    # ========================================
                    with profile_tabs[5]:
                        st.markdown("### 📝 Recruiter Notes")

                        notes_value = st.text_area(
                            "Private recruiter notes",
                            value=state.get("notes", ""),
                            height=180,
                            placeholder=(
                                "Example: Verify cloud experience during screening."
                            ),
                            key=f"profile_notes_{safe_filename(selected_resume)}",
                        )

                        if st.button(
                            "💾 Save Notes",
                            key=f"profile_save_notes_{safe_filename(selected_resume)}",
                            use_container_width=True,
                        ):
                            set_candidate_notes(
                                workflow,
                                selected_resume,
                                notes_value,
                            )
                            persist_workflow_state(
                                selected_resume,
                                workflow[selected_resume],
                            )
                            st.session_state.candidate_workflow = workflow
                            st.success("Recruiter notes saved permanently.")
                            st.rerun()

                    # ========================================
                    # PIPELINE HISTORY
                    # ========================================
                    with profile_tabs[6]:
                        st.markdown("### 🕒 Status History")

                        status_history = get_status_history(
                            st.session_state.job_fingerprint,
                            selected_resume,
                        )

                        if status_history:
                            history_rows = [
                                {
                                    "From": item["old_status"],
                                    "To": item["new_status"],
                                    "Changed At (UTC)": item["changed_at"],
                                }
                                for item in status_history
                            ]
                            st.dataframe(
                                pd.DataFrame(history_rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No status changes recorded yet.")

                        st.markdown("### 📝 Notes History")

                        note_history = get_note_history(
                            st.session_state.job_fingerprint,
                            selected_resume,
                        )

                        if note_history:
                            for index, item in enumerate(note_history, start=1):
                                with st.expander(
                                    f"Note update {index} - {item['changed_at']} UTC",
                                    expanded=(index == 1),
                                ):
                                    st.write(item["notes"] or "Notes cleared.")
                        else:
                            st.info("No note history recorded yet.")

                        st.caption(
                            "Recruiter workflow data is stored locally in recruiter_data.db."
                        )

                    # ========================================
                    # EXPLAINABILITY / EVIDENCE
                    # ========================================
                    with profile_tabs[7]:
                        st.markdown("### 🔍 Why This Match?")
                        st.caption(
                            "A deterministic, evidence-first view of the existing ATS signals. "
                            "It explains the score; it does not create a new hiring score or decision."
                        )

                        try:
                            jd_for_explanation = st.session_state.get(
                                "jd_text",
                                "",
                            )

                            candidate_matrix_for_explanation = (
                                build_candidate_requirement_matrix(
                                    jd_analysis=st.session_state.get(
                                        "jd_analysis",
                                        {},
                                    ),
                                    candidate_requirements=requirements,
                                    candidate_intelligence=intelligence,
                                )
                            )

                            explanation = build_explainability(
                                resume_text=cache.get("text", ""),
                                jd_text=jd_for_explanation,
                                analysis_result=analysis_result,
                                ats_result=ats_result,
                                intelligence=intelligence,
                                quality=quality,
                                requirements=requirements,
                                candidate_matrix=candidate_matrix_for_explanation,
                            )

                            ex1, ex2, ex3, ex4 = st.columns(4)

                            with ex1:
                                st.metric(
                                    "Existing ATS",
                                    f"{explanation['ats_score']:.1f}%",
                                )
                            with ex2:
                                st.metric(
                                    "Required Skills",
                                    f"{len(explanation['required_matched'])}/{explanation['requirement_counts']['required_total']}",
                                )
                            with ex3:
                                st.metric(
                                    "Matched Skills",
                                    len(explanation["matched_skills"]),
                                )
                            with ex4:
                                st.metric(
                                    "Resume Quality",
                                    f"{explanation['resume_quality']:.1f}%",
                                )

                            st.divider()
                            st.markdown("### 📊 Score Breakdown")
                            breakdown_df = pd.DataFrame(
                                explanation["score_breakdown"],
                                columns=[
                                    "component",
                                    "score",
                                    "weight",
                                    "contribution",
                                ],
                            ).rename(
                                columns={
                                    "component": "Component",
                                    "score": "Score (%)",
                                    "weight": "Weight (%)",
                                    "contribution": "Weighted Contribution",
                                }
                            )
                            st.dataframe(
                                breakdown_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Component": st.column_config.TextColumn("Component"),
                                    "Score (%)": st.column_config.NumberColumn("Score (%)", format="%.1f"),
                                    "Weight (%)": st.column_config.NumberColumn("Weight (%)", format="%.0f"),
                                    "Weighted Contribution": st.column_config.NumberColumn("Weighted Contribution", format="%.1f"),
                                },
                            )

                            st.info(
                                "ATS formula used by this project: Skill 35% + Semantic 30% + Lexical 20% + Keyword 15%. "
                                "These are project-defined weights, not an official ATS vendor formula."
                            )

                            left, right = st.columns(2)

                            with left:
                                st.markdown("### ✅ Positive Evidence")
                                if explanation["strengths"]:
                                    for item in explanation["strengths"][:10]:
                                        st.write(f"✅ {item}")
                                else:
                                    st.write("No explicit positive evidence was extracted.")

                                st.markdown("### 🧩 Matched Skills")
                                st.write(
                                    ", ".join(explanation["matched_skills"])
                                    or "None detected"
                                )

                                st.markdown("### 🔑 Matched Keywords")
                                st.write(
                                    ", ".join(explanation["matched_keywords"])
                                    or "None detected"
                                )

                            with right:
                                st.markdown("### ⚠️ Gaps / Verification Areas")
                                if explanation["gaps"]:
                                    for item in explanation["gaps"][:10]:
                                        st.write(f"⚠️ {item}")
                                else:
                                    st.write("No specific gap signal was extracted.")

                                st.markdown("### ❌ Required Skills Missing / Not Detected")
                                st.write(
                                    ", ".join(explanation["required_missing"])
                                    or "None detected"
                                )

                                st.markdown("### ⭐ Preferred Skills Missing / Not Detected")
                                st.write(
                                    ", ".join(explanation["preferred_missing"])
                                    or "None detected"
                                )

                            st.divider()
                            st.markdown("### 📋 Requirement Coverage")
                            required_total = explanation["requirement_counts"]["required_total"]
                            required_matched = explanation["requirement_counts"]["required_matched"]
                            if required_total:
                                st.progress(
                                    max(0.0, min(1.0, required_matched / required_total))
                                )
                                st.caption(
                                    f"Required-skill evidence: {required_matched} / {required_total} signals matched or detected."
                                )
                            else:
                                st.info("The current JD does not expose a required-skill matrix entry.")

                            st.divider()
                            st.markdown("### 🧾 Resume Evidence")
                            if explanation["resume_evidence"]:
                                for idx, sentence in enumerate(
                                    explanation["resume_evidence"],
                                    start=1,
                                ):
                                    with st.expander(
                                        f"Evidence {idx}",
                                        expanded=(idx <= 3),
                                    ):
                                        st.write(sentence)
                            else:
                                st.info(
                                    "No matching evidence sentence was extracted from the supplied resume text. "
                                    "Use the full resume text to verify requirements manually."
                                )

                            st.divider()
                            st.markdown("### 🧠 Signal Interpretation")
                            st.write(
                                f"**Skill Match:** {explanation['signals']['skill_score']:.1f}% — reflects extracted skill overlap."
                            )
                            st.write(
                                f"**Semantic Match:** {explanation['signals']['semantic_score']:.1f}% — aggregate semantic similarity; it is not a per-requirement proof."
                            )
                            st.write(
                                f"**Lexical Match:** {explanation['signals']['lexical_score']:.1f}% — textual overlap signal used by the existing engine."
                            )
                            st.write(
                                f"**Keyword Match:** {explanation['signals']['keyword_score']:.1f}% — keyword overlap from the existing analysis."
                            )

                            with st.expander("ℹ️ Interpretation & limitations", expanded=False):
                                for limitation in explanation["limitations"]:
                                    st.write(f"• {limitation}")

                            st.download_button(
                                "📥 Download Explainability Report",
                                data=explanation_to_text(explanation),
                                file_name=f"{safe_filename(selected_resume)}_explainability.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key=f"explainability_download_{safe_filename(selected_resume)}",
                            )

                        except Exception as error:
                            st.error(
                                f"Explainability view could not be generated: {error}"
                            )



    # ========================================================
    # AI-ASSISTED SCREENING BRIEF TAB
    # ========================================================
    with brief_tab:
        st.subheader("🧠 AI-Assisted Screening Brief")
        st.caption(
            "Convert the existing candidate analysis into a recruiter-ready "
            "summary of strengths, gaps, verification areas and interview questions."
        )

        if not st.session_state.job_fingerprint:
            st.info(
                "Analyze a Job Description and resumes first to activate screening briefs."
            )
        else:
            brief_rows = df.to_dict("records")

            brief_options = {
                str(row.get("Resume", "")): str(
                    row.get("Candidate", "Candidate")
                )
                for row in brief_rows
                if str(row.get("Resume", ""))
            }

            if not brief_options:
                st.warning("No analyzed candidates are available.")
            else:
                selected_brief_resume = st.selectbox(
                    "👤 Select Candidate",
                    list(brief_options.keys()),
                    format_func=lambda x: (
                        f"{brief_options[x]} — {x}"
                    ),
                    key="screening_brief_candidate",
                )

                selected_brief_row = next(
                    (
                        row for row in brief_rows
                        if str(row.get("Resume", "")) == selected_brief_resume
                    ),
                    None,
                )

                cache = st.session_state.resume_cache.get(
                    selected_brief_resume,
                    {},
                )

                if not selected_brief_row or not cache:
                    st.warning(
                        "Detailed candidate analysis is unavailable."
                    )
                else:
                    if st.button(
                        "✨ Generate Screening Brief",
                        type="primary",
                        use_container_width=True,
                        key="generate_screening_brief",
                    ):
                        try:
                            st.session_state.screening_brief[
                                selected_brief_resume
                            ] = build_screening_brief(
                                candidate_row=selected_brief_row,
                                analysis_result=cache.get("analysis", {}),
                                ats_result=cache.get("ats", {}),
                                intelligence=cache.get("intelligence", {}),
                                quality=cache.get("quality", {}),
                                requirements=cache.get("requirements", {}),
                            )
                            st.success(
                                "Screening brief generated from the existing evidence."
                            )
                        except Exception as error:
                            st.error(
                                f"Could not generate screening brief: {error}"
                            )

                    brief = st.session_state.screening_brief.get(
                        selected_brief_resume
                    )


                    # ----------------------------------------
                    # FREE GEMINI-ENHANCED BRIEF
                    # ----------------------------------------
                    st.divider()
                    st.markdown("### ✨ Free Gemini-Enhanced Screening")

                    st.caption(
                        "Optional Gemini synthesis using the Google Gemini Developer API. "
                        "Your deterministic ATS and requirement scores remain unchanged. "
                        "The app uses a free-tier Flash model and can fall back if a model is temporarily unavailable."
                    )

                    if not gemini_is_configured():
                        st.info(
                            "Gemini is not configured. Set GEMINI_API_KEY "
                            "and restart/reload Streamlit."
                        )
                    else:
                        if st.button(
                            "✨ Generate with Free Gemini",
                            type="secondary",
                            use_container_width=True,
                            key="generate_gemini_screening_brief",
                        ):
                            try:
                                with st.spinner(
                                    "Generating evidence-grounded Gemini brief..."
                                ):
                                    gemini_result = (
                                        generate_gemini_screening_brief(
                                            candidate_row=selected_brief_row,
                                            analysis_result=cache.get("analysis", {}),
                                            ats_result=cache.get("ats", {}),
                                            intelligence=cache.get("intelligence", {}),
                                            quality=cache.get("quality", {}),
                                            requirements=cache.get("requirements", {}),
                                            model=GEMINI_MODEL,
                                        )
                                    )

                                st.session_state.gemini_screening_brief[
                                    selected_brief_resume
                                ] = gemini_result

                                st.success(
                                    "Gemini screening brief generated."
                                )

                            except Exception as error:
                                st.error(
                                    f"Could not generate the Gemini brief: {error}"
                                )

                    gemini_brief = (
                        st.session_state.gemini_screening_brief.get(
                            selected_brief_resume
                        )
                    )

                    if gemini_brief:
                        st.markdown("#### 📝 Executive Summary")
                        st.info(
                            gemini_brief.get(
                                "executive_summary",
                                "No summary available.",
                            )
                        )

                        gl, gr = st.columns(2)

                        with gl:
                            st.markdown("#### ✅ Strengths")
                            for item in gemini_brief.get(
                                "strengths",
                                [],
                            ):
                                st.success(item)

                            st.markdown(
                                "#### ⚠️ Gaps / Verification"
                            )
                            for item in gemini_brief.get(
                                "gaps_and_verification",
                                [],
                            ):
                                st.warning(item)

                            st.markdown("#### 🎯 Interview Focus")
                            for item in gemini_brief.get(
                                "interview_focus",
                                [],
                            ):
                                st.info(item)

                        with gr:
                            st.markdown(
                                "#### 💬 Recruiter Talking Points"
                            )
                            for item in gemini_brief.get(
                                "recruiter_talking_points",
                                [],
                            ):
                                st.write(
                                    f"• {item}"
                                )

                            st.markdown(
                                "#### ❓ Suggested Questions"
                            )

                            for number, question in enumerate(
                                gemini_brief.get(
                                    "suggested_questions",
                                    [],
                                ),
                                start=1,
                            ):
                                st.write(
                                    f"**{number}.** {question}"
                                )

                            with st.expander(
                                "🔎 Evidence Notes",
                                expanded=False,
                            ):
                                for item in gemini_brief.get(
                                    "evidence_notes",
                                    [],
                                ):
                                    st.write(
                                        f"• {item}"
                                    )

                                st.caption(
                                    "Source: "
                                    + str(
                                        gemini_brief.get(
                                            "source",
                                            "Google Gemini",
                                        )
                                    )
                                    + " | Model used: "
                                    + str(
                                        gemini_brief.get(
                                            "model",
                                            GEMINI_MODEL,
                                        )
                                    )
                                    + (
                                        " | Free-tier fallback used"
                                        if gemini_brief.get("fallback_used")
                                        else ""
                                    )
                                )

                        st.markdown("#### 📥 Export Gemini Brief")

                        gemini_text = gemini_brief_to_text(
                            gemini_brief
                        )

                        ge1, ge2 = st.columns(2)

                        with ge1:
                            st.download_button(
                                "📄 Download Gemini Brief",
                                data=gemini_text,
                                file_name=(
                                    f"{safe_filename(selected_brief_resume)}_"
                                    "gemini_screening_brief.txt"
                                ),
                                mime="text/plain",
                                use_container_width=True,
                            )

                        with ge2:
                            st.text_area(
                                "Copy-ready Gemini brief",
                                value=gemini_text,
                                height=300,
                                key=(
                                    "gemini_brief_copy_"
                                    f"{safe_filename(selected_brief_resume)}"
                                ),
                            )

                    if brief:
                        metrics = brief.get("metrics", {}) or {}

                        st.divider()

                        m1, m2, m3, m4 = st.columns(4)

                        with m1:
                            st.metric(
                                "ATS",
                                f"{metrics.get('ats_score', 0):.1f}%"
                            )

                        with m2:
                            req = metrics.get("requirement_score")
                            st.metric(
                                "Requirement Match",
                                (
                                    f"{req:.1f}%"
                                    if req is not None
                                    else "N/A"
                                ),
                            )

                        with m3:
                            st.metric(
                                "Resume Quality",
                                f"{metrics.get('quality_score', 0):.1f}%"
                            )

                        with m4:
                            st.metric(
                                "Experience",
                                f"{metrics.get('experience_years', 0):.1f} yrs"
                            )

                        st.info(
                            "📝 " + brief.get("summary", "")
                        )

                        left, right = st.columns(2)

                        with left:
                            st.markdown("### ✅ Key Strengths")
                            strengths = brief.get("strengths", [])
                            if strengths:
                                for item in strengths:
                                    st.success(item)
                            else:
                                st.write("No specific strength signals generated.")

                        with right:
                            st.markdown("### ⚠️ Key Gaps")
                            gaps = brief.get("gaps", [])
                            if gaps:
                                for item in gaps:
                                    st.warning(item)
                            else:
                                st.write("No specific verification gaps generated.")

                        st.divider()

                        st.markdown("### 🎯 Interview Focus Areas")
                        for item in brief.get("interview_focus", []):
                            st.info(item)

                        st.markdown("### 💬 Recruiter Talking Points")
                        for item in brief.get("talking_points", []):
                            st.write(f"• {item}")

                        st.markdown("### ❓ Suggested Interview Questions")

                        for index, question in enumerate(
                            brief.get("interview_questions", []),
                            start=1,
                        ):
                            st.write(f"**{index}.** {question}")

                        with st.expander(
                            "🔎 Evidence Used",
                            expanded=False,
                        ):
                            for item in brief.get("evidence", []):
                                st.write(f"• {item}")

                            st.markdown("### Generated From")
                            for item in brief.get("generated_from", []):
                                st.write(f"• {item}")

                        st.divider()
                        st.markdown("### 📥 Export")

                        text_version = brief_to_text(brief)

                        e1, e2 = st.columns(2)

                        with e1:
                            st.download_button(
                                "📄 Download Screening Brief",
                                data=text_version,
                                file_name=(
                                    f"{safe_filename(selected_brief_resume)}_"
                                    "screening_brief.txt"
                                ),
                                mime="text/plain",
                                use_container_width=True,
                            )

                        with e2:
                            st.text_area(
                                "Copy-ready brief",
                                value=text_version,
                                height=280,
                                key=f"brief_copy_{safe_filename(selected_brief_resume)}",
                            )
                    else:
                        st.info(
                            "Click **Generate Screening Brief** to create the recruiter summary."
                        )


    # ========================================================
    # RAG RECRUITER COPILOT TAB
    # ========================================================
    with copilot_tab:
        st.subheader("🤖 RAG Recruiter Copilot")
        st.caption(
            "Ask questions about a candidate. The app retrieves relevant "
            "resume/JD/interview evidence locally before optional free Gemini synthesis."
        )

        if not st.session_state.job_fingerprint:
            st.info(
                "Analyze a Job Description and resumes first to activate Recruiter Copilot."
            )
        else:
            copilot_rows = df.to_dict("records")

            candidate_options = {
                str(row.get("Resume", "")): str(
                    row.get(
                        "Candidate",
                        "Candidate",
                    )
                )
                for row in copilot_rows
                if str(
                    row.get(
                        "Resume",
                        "",
                    )
                )
            }

            if not candidate_options:
                st.warning(
                    "No analyzed candidates are available."
                )
            else:
                selected_copilot_resume = st.selectbox(
                    "👤 Candidate",
                    list(
                        candidate_options.keys()
                    ),
                    format_func=lambda value: (
                        f"{candidate_options[value]} — {value}"
                    ),
                    key="copilot_candidate_selector",
                )

                selected_copilot_row = next(
                    (
                        row
                        for row in copilot_rows
                        if str(
                            row.get(
                                "Resume",
                                "",
                            )
                        )
                        == selected_copilot_resume
                    ),
                    None,
                )

                copilot_cache = (
                    st.session_state.resume_cache.get(
                        selected_copilot_resume,
                        {},
                    )
                )

                if not selected_copilot_row or not copilot_cache:
                    st.warning(
                        "Candidate evidence is unavailable."
                    )
                else:
                    interview_history = (
                        get_candidate_interview_history(
                            st.session_state.job_fingerprint,
                            selected_copilot_resume,
                        )
                    )

                    communication_history = (
                        list_communications(
                            st.session_state.job_fingerprint,
                            resume_name=selected_copilot_resume,
                        )
                    )

                    deterministic_brief = (
                        st.session_state.screening_brief.get(
                            selected_copilot_resume
                        )
                    )

                    if deterministic_brief is None:
                        try:
                            deterministic_brief = (
                                build_screening_brief(
                                    candidate_row=selected_copilot_row,
                                    analysis_result=copilot_cache.get(
                                        "analysis",
                                        {},
                                    ),
                                    ats_result=copilot_cache.get(
                                        "ats",
                                        {},
                                    ),
                                    intelligence=copilot_cache.get(
                                        "intelligence",
                                        {},
                                    ),
                                    quality=copilot_cache.get(
                                        "quality",
                                        {},
                                    ),
                                    requirements=copilot_cache.get(
                                        "requirements",
                                        {},
                                    ),
                                )
                            )
                        except Exception:
                            deterministic_brief = None

                    corpus = build_copilot_corpus(
                        candidate_row=selected_copilot_row,
                        resume_text=copilot_cache.get(
                            "text",
                            "",
                        ),
                        jd_text=st.session_state.jd_text,
                        analysis_result=copilot_cache.get(
                            "analysis",
                            {},
                        ),
                        intelligence=copilot_cache.get(
                            "intelligence",
                            {},
                        ),
                        quality=copilot_cache.get(
                            "quality",
                            {},
                        ),
                        requirements=copilot_cache.get(
                            "requirements",
                            {},
                        ),
                        screening_brief=deterministic_brief,
                        interview_history=interview_history,
                        communication_history=communication_history,
                    )

                    st.success(
                        f"📚 RAG index ready: {len(corpus)} evidence chunks"
                    )

                    suggested = [
                        "What should I ask about this candidate's strongest project?",
                        "Which required skills should I verify?",
                        "What evidence supports the requirement match?",
                        "What gaps should I clarify?",
                        "What does the interview history tell me?",
                        "What recruiter talking points should I use?",
                    ]

                    suggested_choice = st.selectbox(
                        "💡 Suggested Question",
                        ["Custom question"] + suggested,
                        key="copilot_suggested_choice",
                    )

                    question = st.text_area(
                        "💬 Ask Recruiter Copilot",
                        value=(
                            ""
                            if suggested_choice == "Custom question"
                            else suggested_choice
                        ),
                        height=100,
                        placeholder=(
                            "Example: What evidence supports this candidate's "
                            "Python experience?"
                        ),
                        key="copilot_question",
                    )

                    top_k = st.slider(
                        "Evidence chunks",
                        min_value=3,
                        max_value=8,
                        value=5,
                        key="copilot_top_k",
                    )

                    action1, action2 = st.columns(2)

                    with action1:
                        run_copilot = st.button(
                            "🔎 Retrieve Evidence & Ask Copilot",
                            type="primary",
                            use_container_width=True,
                            key="run_copilot",
                        )

                    with action2:
                        clear_copilot = st.button(
                            "🧹 Clear Candidate Chat",
                            use_container_width=True,
                            key="clear_copilot",
                        )

                    if clear_copilot:
                        st.session_state.copilot_history[
                            selected_copilot_resume
                        ] = []
                        st.rerun()

                    if run_copilot:
                        if not question.strip():
                            st.warning(
                                "Enter a recruiter question first."
                            )
                        else:
                            retrieved = retrieve_evidence(
                                question,
                                corpus,
                                top_k=top_k,
                            )

                            turn = {
                                "question": question.strip(),
                                "retrieved": retrieved,
                                "answer": "",
                                "meta": {},
                            }

                            if retrieved:
                                api_key = os.getenv(
                                    "GEMINI_API_KEY",
                                    "",
                                ).strip()

                                if api_key:
                                    try:
                                        with st.spinner(
                                            "Retrieving evidence and generating grounded Gemini answer..."
                                        ):
                                            answer_data = (
                                                generate_gemini_copilot_answer(
                                                    candidate_name=str(
                                                        selected_copilot_row.get(
                                                            "Candidate",
                                                            "Candidate",
                                                        )
                                                    ),
                                                    question=question.strip(),
                                                    retrieved=retrieved,
                                                    model=GEMINI_COPILOT_MODEL,
                                                )
                                            )

                                        turn["answer"] = (
                                            answer_data.get(
                                                "answer",
                                                "",
                                            )
                                        )
                                        turn["meta"] = answer_data

                                    except Exception as error:
                                        turn["answer"] = (
                                            evidence_only_fallback(
                                                candidate_name=str(
                                                    selected_copilot_row.get(
                                                        "Candidate",
                                                        "Candidate",
                                                    )
                                                ),
                                                retrieved=retrieved,
                                            )
                                        )
                                        turn["meta"] = {
                                            "fallback": True,
                                            "error": str(error),
                                        }

                                        st.warning(
                                            "Gemini was unavailable, so the "
                                            "retrieved evidence is shown instead."
                                        )
                                else:
                                    turn["answer"] = (
                                        evidence_only_fallback(
                                            candidate_name=str(
                                                selected_copilot_row.get(
                                                    "Candidate",
                                                    "Candidate",
                                                )
                                            ),
                                            retrieved=retrieved,
                                        )
                                    )
                                    turn["meta"] = {
                                        "fallback": True,
                                        "reason": (
                                            "GEMINI_API_KEY not configured"
                                        ),
                                    }
                            else:
                                turn["answer"] = (
                                    "No relevant evidence was retrieved. "
                                    "Try a more specific question."
                                )

                            history = st.session_state.copilot_history.get(
                                selected_copilot_resume,
                                [],
                            )

                            history = history + [turn]

                            st.session_state.copilot_history[
                                selected_copilot_resume
                            ] = history[-10:]

                    history = st.session_state.copilot_history.get(
                        selected_copilot_resume,
                        [],
                    )

                    if history:
                        latest = history[-1]

                        st.divider()
                        st.markdown("### 🧠 Copilot Answer")
                        st.info(
                            latest.get(
                                "answer",
                                "No answer available.",
                            )
                        )

                        meta = latest.get(
                            "meta",
                            {},
                        ) or {}

                        if meta.get("model"):
                            st.caption(
                                f"Source: {meta.get('source', 'Google Gemini')} "
                                f"| Model: {meta.get('model', GEMINI_COPILOT_MODEL)}"
                                + (
                                    " | fallback used"
                                    if meta.get(
                                        "fallback_used"
                                    )
                                    else ""
                                )
                            )

                        st.markdown("### 📚 Retrieved Evidence")

                        retrieved = latest.get(
                            "retrieved",
                            [],
                        )

                        for index, item in enumerate(
                            retrieved,
                            start=1,
                        ):
                            with st.expander(
                                (
                                    f"{index}. "
                                    f"{item.get('source')} — "
                                    f"{item.get('section')} "
                                    f"(retrieval {item.get('retrieval_score', 0):.3f})"
                                ),
                                expanded=(index == 1),
                            ):
                                st.write(
                                    item.get(
                                        "text",
                                        "",
                                    )
                                )
                                st.caption(
                                    f"TF-IDF: {item.get('tfidf_score', 0):.3f} "
                                    f"| Keyword overlap: {item.get('keyword_overlap', 0):.3f}"
                                )

                        with st.expander(
                            "🔐 Why this is RAG",
                            expanded=False,
                        ):
                            st.write(
                                "1. The question is converted into a retrieval query."
                            )
                            st.write(
                                "2. Local TF-IDF + keyword overlap selects the most relevant evidence."
                            )
                            st.write(
                                "3. Only the selected evidence chunks are passed to Gemini."
                            )
                            st.write(
                                "4. The answer and retrieved sources are shown together."
                            )
                            st.write(
                                "5. ATS and requirement scores are not recomputed by the copilot."
                            )

                        st.markdown("### 🕒 Copilot Conversation")

                        for turn_index, turn in enumerate(
                            reversed(history),
                            start=1,
                        ):
                            with st.expander(
                                f"Question {len(history) - turn_index + 1}: "
                                f"{turn.get('question', '')}",
                                expanded=(turn_index == 1),
                            ):
                                st.write(
                                    turn.get(
                                        "answer",
                                        "",
                                    )
                                )

                        export_lines = [
                            "RAG RECRUITER COPILOT",
                            "=" * 55,
                            (
                                "Candidate: "
                                + str(
                                    selected_copilot_row.get(
                                        "Candidate",
                                        "Candidate",
                                    )
                                )
                            ),
                            "",
                        ]

                        for index, turn in enumerate(
                            history,
                            start=1,
                        ):
                            export_lines.extend(
                                [
                                    f"QUESTION {index}",
                                    "-" * 20,
                                    turn.get(
                                        "question",
                                        "",
                                    ),
                                    "",
                                    "ANSWER",
                                    "-" * 20,
                                    turn.get(
                                        "answer",
                                        "",
                                    ),
                                    "",
                                    "EVIDENCE",
                                    "-" * 20,
                                ]
                            )

                            for source_index, item in enumerate(
                                turn.get(
                                    "retrieved",
                                    [],
                                ),
                                start=1,
                            ):
                                export_lines.append(
                                    (
                                        f"[{source_index}] "
                                        f"{item.get('source')} / "
                                        f"{item.get('section')}: "
                                        f"{item.get('text')}"
                                    )
                                )

                            export_lines.append("")

                        st.download_button(
                            "📄 Download Copilot Conversation",
                            data="\n".join(
                                export_lines
                            ),
                            file_name=(
                                f"{safe_filename(selected_copilot_resume)}_"
                                "rag_copilot.txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "Ask a question to start the recruiter copilot."
                        )


    # ========================================================
    # CANDIDATE COMPARISON TAB
    # ========================================================
    with comparison_tab:
        st.subheader("⚖️ Compare Candidates")
        st.caption(
            "Compare two analyzed candidates side by side using the same "
            "project-defined signals and extracted evidence."
        )

        if not st.session_state.job_fingerprint:
            st.info(
                "Analyze a Job Description and resumes first to activate candidate comparison."
            )
        else:
            comparison_rows = df.to_dict("records")

            comparison_options = {
                str(row.get("Resume", "")): str(
                    row.get(
                        "Candidate",
                        "Candidate",
                    )
                )
                for row in comparison_rows
                if str(row.get("Resume", ""))
            }

            if len(comparison_options) < 2:
                st.info(
                    "Upload and analyze at least two resumes to compare candidates."
                )
            else:
                option_keys = list(
                    comparison_options.keys()
                )

                left_col, right_col = st.columns(2)

                with left_col:
                    resume_a = st.selectbox(
                        "👤 Candidate A",
                        option_keys,
                        format_func=lambda value: (
                            f"{comparison_options[value]} — {value}"
                        ),
                        key="comparison_candidate_a",
                    )

                remaining_keys = [
                    key
                    for key in option_keys
                    if key != resume_a
                ]

                with right_col:
                    default_b_index = 0

                    current_b = st.session_state.get(
                        "comparison_candidate_b"
                    )

                    if (
                        current_b in remaining_keys
                    ):
                        default_b_index = remaining_keys.index(
                            current_b
                        )

                    resume_b = st.selectbox(
                        "👤 Candidate B",
                        remaining_keys,
                        index=default_b_index,
                        format_func=lambda value: (
                            f"{comparison_options[value]} — {value}"
                        ),
                        key="comparison_candidate_b",
                    )

                row_a = next(
                    (
                        row
                        for row in comparison_rows
                        if str(
                            row.get(
                                "Resume",
                                "",
                            )
                        ) == resume_a
                    ),
                    None,
                )

                row_b = next(
                    (
                        row
                        for row in comparison_rows
                        if str(
                            row.get(
                                "Resume",
                                "",
                            )
                        ) == resume_b
                    ),
                    None,
                )

                cache_a = st.session_state.resume_cache.get(
                    resume_a,
                    {},
                )

                cache_b = st.session_state.resume_cache.get(
                    resume_b,
                    {},
                )

                if not row_a or not row_b or not cache_a or not cache_b:
                    st.warning(
                        "Detailed candidate analysis is unavailable for one or both candidates."
                    )
                else:
                    comparison = build_candidate_comparison(
                        row_a=row_a,
                        row_b=row_b,
                        cache_a=cache_a,
                        cache_b=cache_b,
                    )

                    st.divider()

                    st.markdown("### 📊 Side-by-Side Metrics")

                    metric_rows = []

                    for label, values in (
                        comparison.get(
                            "metrics",
                            {},
                        )
                        or {}
                    ).items():
                        metric_rows.append(
                            {
                                "Metric": label,
                                comparison["candidate_a"]["name"]: (
                                    values.get("candidate_a")
                                ),
                                comparison["candidate_b"]["name"]: (
                                    values.get("candidate_b")
                                ),
                                "Difference (A-B)": (
                                    values.get(
                                        "difference_a_minus_b"
                                    )
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(
                            metric_rows
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.divider()

                    # ----------------------------------------
                    # Skills
                    # ----------------------------------------
                    st.markdown("### 🛠️ Skill Comparison")

                    skills = comparison.get(
                        "skills",
                        {},
                    ) or {}

                    skill_left, skill_middle, skill_right = st.columns(3)

                    with skill_left:
                        st.markdown("#### 🔗 Shared Skills")

                        shared = skills.get(
                            "shared",
                            [],
                        )

                        if shared:
                            for skill in shared:
                                st.write(f"• {skill}")
                        else:
                            st.info(
                                "No shared extracted skills."
                            )

                    with skill_middle:
                        st.markdown(
                            "#### 👤 Only Candidate A"
                        )

                        only_a = skills.get(
                            "only_a",
                            [],
                        )

                        if only_a:
                            for skill in only_a:
                                st.write(f"• {skill}")
                        else:
                            st.info(
                                "No unique extracted skills."
                            )

                    with skill_right:
                        st.markdown(
                            "#### 👤 Only Candidate B"
                        )

                        only_b = skills.get(
                            "only_b",
                            [],
                        )

                        if only_b:
                            for skill in only_b:
                                st.write(f"• {skill}")
                        else:
                            st.info(
                                "No unique extracted skills."
                            )

                    # ----------------------------------------
                    # Required / preferred skill coverage
                    # ----------------------------------------
                    st.divider()
                    st.markdown(
                        "### 🎯 Requirement Coverage"
                    )

                    req_a = comparison.get(
                        "required_skills",
                        {},
                    ).get(
                        "a",
                        {},
                    )

                    req_b = comparison.get(
                        "required_skills",
                        {},
                    ).get(
                        "b",
                        {},
                    )

                    pref_a = comparison.get(
                        "preferred_skills",
                        {},
                    ).get(
                        "a",
                        {},
                    )

                    pref_b = comparison.get(
                        "preferred_skills",
                        {},
                    ).get(
                        "b",
                        {},
                    )

                    r1, r2 = st.columns(2)

                    with r1:
                        st.markdown(
                            f"#### {comparison['candidate_a']['name']}"
                        )

                        st.markdown(
                            "**Required skills matched**"
                        )
                        st.write(
                            ", ".join(
                                req_a.get(
                                    "matched",
                                    [],
                                )
                            )
                            or "None detected"
                        )

                        st.markdown(
                            "**Required skills missing**"
                        )
                        st.write(
                            ", ".join(
                                req_a.get(
                                    "missing",
                                    [],
                                )
                            )
                            or "None detected"
                        )

                        st.markdown(
                            "**Preferred skills matched**"
                        )
                        st.write(
                            ", ".join(
                                pref_a.get(
                                    "matched",
                                    [],
                                )
                            )
                            or "None detected"
                        )

                    with r2:
                        st.markdown(
                            f"#### {comparison['candidate_b']['name']}"
                        )

                        st.markdown(
                            "**Required skills matched**"
                        )
                        st.write(
                            ", ".join(
                                req_b.get(
                                    "matched",
                                    [],
                                )
                            )
                            or "None detected"
                        )

                        st.markdown(
                            "**Required skills missing**"
                        )
                        st.write(
                            ", ".join(
                                req_b.get(
                                    "missing",
                                    [],
                                )
                            )
                            or "None detected"
                        )

                        st.markdown(
                            "**Preferred skills matched**"
                        )
                        st.write(
                            ", ".join(
                                pref_b.get(
                                    "matched",
                                    [],
                                )
                            )
                            or "None detected"
                        )

                    # ----------------------------------------
                    # Candidate details
                    # ----------------------------------------
                    st.divider()
                    st.markdown(
                        "### 👥 Candidate Evidence"
                    )

                    detail_a, detail_b = st.columns(2)

                    intelligence_a = cache_a.get(
                        "intelligence",
                        {},
                    ) or {}

                    intelligence_b = cache_b.get(
                        "intelligence",
                        {},
                    ) or {}

                    with detail_a:
                        st.markdown(
                            f"#### {comparison['candidate_a']['name']}"
                        )

                        st.write(
                            "**Experience:** "
                            f"{intelligence_a.get('experience_years', 0)} years"
                        )

                        st.write(
                            "**Education:** "
                            + str(
                                intelligence_a.get(
                                    "education",
                                    "Not detected",
                                )
                                or "Not detected"
                            )
                        )

                        st.write(
                            "**Projects:** "
                            + str(
                                intelligence_a.get(
                                    "projects",
                                    "Not detected",
                                )
                                or "Not detected"
                            )
                        )

                        st.write(
                            "**Certifications:** "
                            + str(
                                intelligence_a.get(
                                    "certifications",
                                    "Not detected",
                                )
                                or "Not detected"
                            )
                        )

                    with detail_b:
                        st.markdown(
                            f"#### {comparison['candidate_b']['name']}"
                        )

                        st.write(
                            "**Experience:** "
                            f"{intelligence_b.get('experience_years', 0)} years"
                        )

                        st.write(
                            "**Education:** "
                            + str(
                                intelligence_b.get(
                                    "education",
                                    "Not detected",
                                )
                                or "Not detected"
                            )
                        )

                        st.write(
                            "**Projects:** "
                            + str(
                                intelligence_b.get(
                                    "projects",
                                    "Not detected",
                                )
                                or "Not detected"
                            )
                        )

                        st.write(
                            "**Certifications:** "
                            + str(
                                intelligence_b.get(
                                    "certifications",
                                    "Not detected",
                                )
                                or "Not detected"
                            )
                        )

                    # ----------------------------------------
                    # Comparison Copilot
                    # ----------------------------------------
                    st.divider()
                    st.markdown(
                        "### 🤖 Comparison Copilot"
                    )

                    st.caption(
                        "Ask a question about both candidates. "
                        "Local RAG retrieves evidence for each candidate before "
                        "optional free Gemini synthesis."
                    )

                    comparison_questions = [
                        "Compare the candidates' evidence for the required skills.",
                        "What are the main evidence-backed differences between these candidates?",
                        "Which skill gaps should I verify for each candidate?",
                        "What should I ask each candidate about their strongest project?",
                        "Compare the candidates' experience and project evidence.",
                    ]

                    comparison_prompt_choice = st.selectbox(
                        "💡 Suggested comparison question",
                        ["Custom question"] + comparison_questions,
                        key="comparison_prompt_choice",
                    )

                    comparison_question = st.text_area(
                        "💬 Ask about both candidates",
                        value=(
                            ""
                            if comparison_prompt_choice == "Custom question"
                            else comparison_prompt_choice
                        ),
                        height=100,
                        key="comparison_question",
                        placeholder=(
                            "Example: Compare the evidence supporting each candidate's "
                            "Python and machine-learning experience."
                        ),
                    )

                    comparison_top_k = st.slider(
                        "Evidence chunks per candidate",
                        min_value=2,
                        max_value=6,
                        value=4,
                        key="comparison_top_k",
                    )

                    run_comparison = st.button(
                        "🔎 Retrieve Both Candidates & Compare",
                        type="primary",
                        use_container_width=True,
                        key="run_comparison_copilot",
                    )

                    if run_comparison:
                        if not comparison_question.strip():
                            st.warning(
                                "Enter a comparison question first."
                            )
                        else:
                            # Build per-candidate RAG corpora.
                            brief_a = st.session_state.screening_brief.get(
                                resume_a
                            )
                            brief_b = st.session_state.screening_brief.get(
                                resume_b
                            )

                            interviews_a = get_candidate_interview_history(
                                st.session_state.job_fingerprint,
                                resume_a,
                            )

                            interviews_b = get_candidate_interview_history(
                                st.session_state.job_fingerprint,
                                resume_b,
                            )

                            communications_a = list_communications(
                                st.session_state.job_fingerprint,
                                resume_name=resume_a,
                            )

                            communications_b = list_communications(
                                st.session_state.job_fingerprint,
                                resume_name=resume_b,
                            )

                            corpus_a = build_copilot_corpus(
                                candidate_row=row_a,
                                resume_text=cache_a.get(
                                    "text",
                                    "",
                                ),
                                jd_text=st.session_state.jd_text,
                                analysis_result=cache_a.get(
                                    "analysis",
                                    {},
                                ),
                                intelligence=cache_a.get(
                                    "intelligence",
                                    {},
                                ),
                                quality=cache_a.get(
                                    "quality",
                                    {},
                                ),
                                requirements=cache_a.get(
                                    "requirements",
                                    {},
                                ),
                                screening_brief=brief_a,
                                interview_history=interviews_a,
                                communication_history=communications_a,
                            )

                            corpus_b = build_copilot_corpus(
                                candidate_row=row_b,
                                resume_text=cache_b.get(
                                    "text",
                                    "",
                                ),
                                jd_text=st.session_state.jd_text,
                                analysis_result=cache_b.get(
                                    "analysis",
                                    {},
                                ),
                                intelligence=cache_b.get(
                                    "intelligence",
                                    {},
                                ),
                                quality=cache_b.get(
                                    "quality",
                                    {},
                                ),
                                requirements=cache_b.get(
                                    "requirements",
                                    {},
                                ),
                                screening_brief=brief_b,
                                interview_history=interviews_b,
                                communication_history=communications_b,
                            )

                            retrieved_a = retrieve_evidence(
                                comparison_question,
                                corpus_a,
                                top_k=comparison_top_k,
                            )

                            retrieved_b = retrieve_evidence(
                                comparison_question,
                                corpus_b,
                                top_k=comparison_top_k,
                            )

                            combined_evidence = (
                                "[CANDIDATE A]\n"
                                + build_comparison_evidence(
                                    comparison=comparison,
                                    row_a=row_a,
                                    row_b=row_b,
                                    cache_a=cache_a,
                                    cache_b=cache_b,
                                )
                                + "\n\n"
                                + "[RETRIEVED A]\n"
                                + "\n".join(
                                    item.get(
                                        "text",
                                        "",
                                    )
                                    for item in retrieved_a
                                )
                                + "\n\n[RETRIEVED B]\n"
                                + "\n".join(
                                    item.get(
                                        "text",
                                        "",
                                    )
                                    for item in retrieved_b
                                )
                            )

                            turn = {
                                "question": comparison_question.strip(),
                                "retrieved_a": retrieved_a,
                                "retrieved_b": retrieved_b,
                                "answer": "",
                                "comparison_evidence": combined_evidence,
                            }

                            api_key = os.getenv(
                                "GEMINI_API_KEY",
                                "",
                            ).strip()

                            if api_key:
                                try:
                                    comparison_prompt = (
                                        "You are a recruiter comparison copilot. "
                                        "Compare Candidate A and Candidate B using ONLY "
                                        "the supplied evidence. Do not declare a winner, "
                                        "ranking, or hiring decision. Do not invent facts. "
                                        "State when evidence is missing. Keep the answer "
                                        "concise and cite whether each observation came "
                                        "from Candidate A, Candidate B, the JD, or other "
                                        "retrieved evidence.\n\n"
                                        f"Recruiter question:\n"
                                        f"{comparison_question.strip()}\n\n"
                                        "Evidence:\n"
                                        + combined_evidence
                                    )

                                    # Reuse the existing Gemini copilot function
                                    # with the combined retrieved evidence.
                                    combined_retrieved = (
                                        [
                                            {
                                                "source": "Candidate A",
                                                "section": item.get(
                                                    "section",
                                                    "",
                                                ),
                                                "text": item.get(
                                                    "text",
                                                    "",
                                                ),
                                                "retrieval_score": item.get(
                                                    "retrieval_score",
                                                    0,
                                                ),
                                                "tfidf_score": item.get(
                                                    "tfidf_score",
                                                    0,
                                                ),
                                                "keyword_overlap": item.get(
                                                    "keyword_overlap",
                                                    0,
                                                ),
                                            }
                                            for item in retrieved_a
                                        ]
                                        + [
                                            {
                                                "source": "Candidate B",
                                                "section": item.get(
                                                    "section",
                                                    "",
                                                ),
                                                "text": item.get(
                                                    "text",
                                                    "",
                                                ),
                                                "retrieval_score": item.get(
                                                    "retrieval_score",
                                                    0,
                                                ),
                                                "tfidf_score": item.get(
                                                    "tfidf_score",
                                                    0,
                                                ),
                                                "keyword_overlap": item.get(
                                                    "keyword_overlap",
                                                    0,
                                                ),
                                            }
                                            for item in retrieved_b
                                        ]
                                    )

                                    with st.spinner(
                                        "Comparing retrieved evidence with Gemini..."
                                    ):
                                        generated = generate_gemini_copilot_answer(
                                            candidate_name=(
                                                f"{comparison['candidate_a']['name']} "
                                                f"vs {comparison['candidate_b']['name']}"
                                            ),
                                            question=(
                                                comparison_prompt
                                            ),
                                            retrieved=combined_retrieved,
                                            model=GEMINI_COPILOT_MODEL,
                                        )

                                    turn["answer"] = generated.get(
                                        "answer",
                                        "",
                                    )

                                    turn["meta"] = generated

                                except Exception as error:
                                    turn["answer"] = (
                                        "Gemini was unavailable. Review the "
                                        "retrieved evidence for each candidate below."
                                    )
                                    turn["meta"] = {
                                        "fallback": True,
                                        "error": str(error),
                                    }
                            else:
                                turn["answer"] = (
                                    "Gemini is not configured. Review the retrieved "
                                    "evidence for each candidate below."
                                )
                                turn["meta"] = {
                                    "fallback": True,
                                    "reason": "GEMINI_API_KEY not configured",
                                }

                            comparison_key = (
                                f"{resume_a}__VS__{resume_b}"
                            )

                            st.session_state.comparison_history[
                                comparison_key
                            ] = turn

                    comparison_key = (
                        f"{resume_a}__VS__{resume_b}"
                    )

                    comparison_turn = (
                        st.session_state.comparison_history.get(
                            comparison_key
                        )
                    )

                    if comparison_turn:
                        st.divider()

                        st.markdown(
                            "### 🧠 Comparison Copilot Answer"
                        )

                        st.info(
                            comparison_turn.get(
                                "answer",
                                "",
                            )
                        )

                        meta = comparison_turn.get(
                            "meta",
                            {},
                        ) or {}

                        if meta.get("model"):
                            st.caption(
                                f"Source: {meta.get('source', 'Google Gemini')} "
                                f"| Model: {meta.get('model', GEMINI_COPILOT_MODEL)}"
                                + (
                                    " | fallback used"
                                    if meta.get(
                                        "fallback_used"
                                    )
                                    else ""
                                )
                            )

                        st.markdown(
                            "### 📚 Retrieved Evidence — Candidate A"
                        )

                        for index, item in enumerate(
                            comparison_turn.get(
                                "retrieved_a",
                                [],
                            ),
                            start=1,
                        ):
                            with st.expander(
                                (
                                    f"A{index}. "
                                    f"{item.get('source')} — "
                                    f"{item.get('section')} "
                                    f"(retrieval {item.get('retrieval_score', 0):.3f})"
                                ),
                                expanded=(index == 1),
                            ):
                                st.write(
                                    item.get(
                                        "text",
                                        "",
                                    )
                                )

                        st.markdown(
                            "### 📚 Retrieved Evidence — Candidate B"
                        )

                        for index, item in enumerate(
                            comparison_turn.get(
                                "retrieved_b",
                                [],
                            ),
                            start=1,
                        ):
                            with st.expander(
                                (
                                    f"B{index}. "
                                    f"{item.get('source')} — "
                                    f"{item.get('section')} "
                                    f"(retrieval {item.get('retrieval_score', 0):.3f})"
                                ),
                                expanded=(index == 1),
                            ):
                                st.write(
                                    item.get(
                                        "text",
                                        "",
                                    )
                                )

                        st.download_button(
                            "📄 Download Comparison Report",
                            data=comparison_to_text(
                                comparison
                            )
                            + "\n\n"
                            + "COPILOT QUESTION\n"
                            + "-" * 30
                            + "\n"
                            + str(
                                comparison_turn.get(
                                    "question",
                                    "",
                                )
                            )
                            + "\n\n"
                            + "COPILOT ANSWER\n"
                            + "-" * 30
                            + "\n"
                            + str(
                                comparison_turn.get(
                                    "answer",
                                    "",
                                )
                            ),
                            file_name="candidate_comparison.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

    # ========================================================
    # INTERVIEW MANAGEMENT TAB
    # ========================================================
    with interview_tab:
        st.subheader("🗓️ Interview Management")
        st.caption(
            "Schedule interviews, track rounds, record scorecards, "
            "capture feedback and keep a persistent interview history."
        )

        if not st.session_state.job_fingerprint:
            st.info(
                "Analyze a Job Description and resumes first to activate interview management."
            )
        else:
            interview_stats = get_interview_statistics(
                st.session_state.job_fingerprint
            )

            i1, i2, i3, i4, i5 = st.columns(5)

            with i1:
                st.metric("Total Interviews", interview_stats["total"])
            with i2:
                st.metric("Scheduled", interview_stats["scheduled"])
            with i3:
                st.metric("Completed", interview_stats["completed"])
            with i4:
                st.metric("Pending Feedback", interview_stats["pending_feedback"])
            with i5:
                st.metric("Proceed", interview_stats["proceed"])

            st.divider()

            interview_candidates = df.to_dict("records")

            interview_candidate_options = {
                str(item.get("Resume", "")): str(
                    item.get("Candidate", "Candidate")
                )
                for item in interview_candidates
                if str(item.get("Resume", ""))
            }

            if not interview_candidate_options:
                st.warning("No analyzed candidates are available.")
            else:
                selected_interview_resume = st.selectbox(
                    "👤 Select Candidate",
                    list(interview_candidate_options.keys()),
                    format_func=lambda value: (
                        f"{interview_candidate_options[value]} — {value}"
                    ),
                    key="interview_candidate_selector",
                )

                selected_interview_row = next(
                    (
                        item
                        for item in interview_candidates
                        if str(item.get("Resume", "")) == selected_interview_resume
                    ),
                    None,
                )

                selected_interview_name = (
                    str(
                        selected_interview_row.get(
                            "Candidate",
                            "Candidate",
                        )
                    )
                    if selected_interview_row
                    else interview_candidate_options[selected_interview_resume]
                )

                candidate_interviews = get_candidate_interview_history(
                    st.session_state.job_fingerprint,
                    selected_interview_resume,
                )

                # ------------------------------------------------
                # Candidate interview summary
                # ------------------------------------------------
                if selected_interview_row:
                    sc1, sc2, sc3, sc4 = st.columns(4)

                    with sc1:
                        st.metric(
                            "ATS",
                            f"{float(selected_interview_row.get('ATS Score', 0) or 0):.1f}%",
                        )

                    with sc2:
                        requirement_value = selected_interview_row.get(
                            "Requirement Match"
                        )
                        st.metric(
                            "Requirement Match",
                            (
                                f"{float(requirement_value):.1f}%"
                                if requirement_value is not None
                                else "N/A"
                            ),
                        )

                    with sc3:
                        st.metric(
                            "Experience",
                            f"{float(selected_interview_row.get('Experience Years', 0) or 0):.1f} yrs",
                        )

                    with sc4:
                        st.metric(
                            "Existing Interviews",
                            len(candidate_interviews),
                        )

                st.divider()

                # ------------------------------------------------
                # Schedule Interview
                # ------------------------------------------------
                st.markdown("### ➕ Schedule New Interview")

                form1, form2 = st.columns(2)

                with form1:
                    interview_round = st.selectbox(
                        "Interview Round",
                        INTERVIEW_ROUNDS,
                        key="new_interview_round",
                    )

                    interview_type = st.selectbox(
                        "Interview Type",
                        INTERVIEW_TYPES,
                        key="new_interview_type",
                    )

                    interview_date = st.date_input(
                        "Interview Date",
                        key="new_interview_date",
                    )

                    interview_time = st.time_input(
                        "Interview Time",
                        key="new_interview_time",
                    )

                with form2:
                    duration = st.number_input(
                        "Duration (minutes)",
                        min_value=15,
                        max_value=480,
                        value=60,
                        step=15,
                        key="new_interview_duration",
                    )

                    interviewer = st.text_input(
                        "Interviewer",
                        placeholder="e.g. Technical Lead",
                        key="new_interview_interviewer",
                    )

                    meeting_link = st.text_input(
                        "Meeting Link",
                        placeholder="https://meet.example.com/...",
                        key="new_interview_link",
                    )

                if st.button(
                    "📅 Schedule Interview",
                    type="primary",
                    use_container_width=True,
                    key="schedule_interview_btn",
                ):
                    try:
                        interview_id = create_interview(
                            job_fingerprint=st.session_state.job_fingerprint,
                            resume_name=selected_interview_resume,
                            candidate_name=selected_interview_name,
                            interview_round=interview_round,
                            interview_type=interview_type,
                            scheduled_date=interview_date.isoformat(),
                            scheduled_time=interview_time.strftime("%H:%M"),
                            duration_minutes=int(duration),
                            interviewer=interviewer,
                            meeting_link=meeting_link,
                        )

                        st.session_state.interview_selected_id = interview_id
                        st.success(
                            f"Interview #{interview_id} scheduled for {selected_interview_name}."
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not schedule interview: {error}")

                st.divider()

                # ------------------------------------------------
                # Interview History / Management
                # ------------------------------------------------
                st.markdown("### 📋 Candidate Interview History")

                if not candidate_interviews:
                    st.info(
                        f"No interviews scheduled for {selected_interview_name} yet."
                    )
                else:
                    history_rows = []

                    for item in candidate_interviews:
                        history_rows.append(
                            {
                                "ID": item["id"],
                                "Round": item["interview_round"],
                                "Type": item["interview_type"],
                                "Date": item["scheduled_date"],
                                "Time": item["scheduled_time"],
                                "Duration": f"{item['duration_minutes']} min",
                                "Interviewer": item["interviewer"] or "Not assigned",
                                "Status": item["status"],
                                "Score": (
                                    f"{float(item['overall_score']):.2f}/5"
                                    if item["overall_score"] is not None
                                    else "Pending"
                                ),
                                "Recommendation": item["recommendation"],
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(history_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    interview_ids = [
                        int(item["id"])
                        for item in candidate_interviews
                    ]

                    default_interview_id = st.session_state.get(
                        "interview_selected_id"
                    )

                    if (
                        default_interview_id not in interview_ids
                    ):
                        default_interview_id = interview_ids[0]

                    selected_interview_id = st.selectbox(
                        "Manage Interview",
                        interview_ids,
                        index=interview_ids.index(default_interview_id),
                        key="managed_interview_id",
                    )

                    st.session_state.interview_selected_id = selected_interview_id

                    interview = get_interview(
                        selected_interview_id
                    )

                    if interview:
                        detail1, detail2 = st.columns(2)

                        with detail1:
                            st.markdown("### 📌 Interview Details")

                            st.write(
                                f"**Round:** {interview['interview_round']}"
                            )
                            st.write(
                                f"**Type:** {interview['interview_type']}"
                            )
                            st.write(
                                f"**Date:** {interview['scheduled_date']}"
                            )
                            st.write(
                                f"**Time:** {interview['scheduled_time']}"
                            )
                            st.write(
                                f"**Duration:** {interview['duration_minutes']} minutes"
                            )
                            st.write(
                                f"**Interviewer:** {interview['interviewer'] or 'Not assigned'}"
                            )

                            if interview.get("meeting_link"):
                                st.link_button(
                                    "🔗 Open Meeting",
                                    interview["meeting_link"],
                                )

                            try:
                                ics_data = generate_ics_event(interview)
                                st.download_button(
                                    "🗓️ Add to Calendar (.ics)",
                                    data=ics_data,
                                    file_name=(
                                        f"interview_{interview['id']}.ics"
                                    ),
                                    mime="text/calendar",
                                    use_container_width=True,
                                )
                            except Exception as error:
                                st.warning(
                                    f"Calendar export unavailable: {error}"
                                )

                        with detail2:
                            st.markdown("### ✏️ Update Interview")

                            updated_status = st.selectbox(
                                "Status",
                                INTERVIEW_STATUSES,
                                index=(
                                    INTERVIEW_STATUSES.index(
                                        interview["status"]
                                    )
                                    if interview["status"] in INTERVIEW_STATUSES
                                    else 0
                                ),
                                key=f"interview_status_{selected_interview_id}",
                            )

                            updated_recommendation = st.selectbox(
                                "Recommendation",
                                RECOMMENDATION_OPTIONS,
                                index=(
                                    RECOMMENDATION_OPTIONS.index(
                                        interview["recommendation"]
                                    )
                                    if interview["recommendation"]
                                    in RECOMMENDATION_OPTIONS
                                    else 0
                                ),
                                key=f"interview_recommendation_{selected_interview_id}",
                            )

                            updated_date = st.date_input(
                                "Date",
                                value=pd.to_datetime(
                                    interview["scheduled_date"]
                                ).date(),
                                key=f"interview_date_{selected_interview_id}",
                            )

                            current_time = interview["scheduled_time"]
                            try:
                                parsed_time = pd.to_datetime(
                                    current_time
                                ).time()
                            except Exception:
                                parsed_time = None

                            updated_time = st.time_input(
                                "Time",
                                value=parsed_time,
                                key=f"interview_time_{selected_interview_id}",
                            )

                            updated_interviewer = st.text_input(
                                "Interviewer",
                                value=interview["interviewer"] or "",
                                key=f"interview_interviewer_{selected_interview_id}",
                            )

                            updated_link = st.text_input(
                                "Meeting Link",
                                value=interview["meeting_link"] or "",
                                key=f"interview_link_{selected_interview_id}",
                            )

                            if st.button(
                                "💾 Save Interview Details",
                                use_container_width=True,
                                key=f"save_interview_{selected_interview_id}",
                            ):
                                try:
                                    update_interview(
                                        selected_interview_id,
                                        status=updated_status,
                                        recommendation=updated_recommendation,
                                        scheduled_date=updated_date.isoformat(),
                                        scheduled_time=updated_time.strftime("%H:%M"),
                                        interviewer=updated_interviewer,
                                        meeting_link=updated_link,
                                    )
                                    st.success(
                                        "Interview details saved permanently."
                                    )
                                    st.rerun()
                                except Exception as error:
                                    st.error(
                                        f"Could not update interview: {error}"
                                    )

                        st.divider()

                        # ------------------------------------------------
                        # Scorecard
                        # ------------------------------------------------
                        st.markdown("### 🧑‍💼 Interview Scorecard")

                        score1, score2, score3 = st.columns(3)

                        def safe_score_value(value):
                            try:
                                return float(value)
                            except (TypeError, ValueError):
                                return 0.0

                        with score1:
                            technical_score = st.slider(
                                "Technical Skills",
                                0.0,
                                5.0,
                                safe_score_value(
                                    interview["technical_score"]
                                ),
                                0.5,
                                key=f"technical_score_{selected_interview_id}",
                            )

                        with score2:
                            communication_score = st.slider(
                                "Communication",
                                0.0,
                                5.0,
                                safe_score_value(
                                    interview["communication_score"]
                                ),
                                0.5,
                                key=f"communication_score_{selected_interview_id}",
                            )

                        with score3:
                            problem_solving_score = st.slider(
                                "Problem Solving",
                                0.0,
                                5.0,
                                safe_score_value(
                                    interview["problem_solving_score"]
                                ),
                                0.5,
                                key=f"problem_score_{selected_interview_id}",
                            )

                        feedback = st.text_area(
                            "Interview Feedback",
                            value=interview["feedback"] or "",
                            height=160,
                            placeholder=(
                                "Record strengths, weaknesses, technical discussion, "
                                "communication observations and follow-up points."
                            ),
                            key=f"interview_feedback_{selected_interview_id}",
                        )

                        internal_notes = st.text_area(
                            "Internal Recruiter Notes",
                            value=interview["internal_notes"] or "",
                            height=120,
                            placeholder="Private notes for the recruiting team...",
                            key=f"interview_internal_{selected_interview_id}",
                        )

                        if st.button(
                            "💾 Save Scorecard & Feedback",
                            type="primary",
                            use_container_width=True,
                            key=f"save_scorecard_{selected_interview_id}",
                        ):
                            try:
                                update_interview(
                                    selected_interview_id,
                                    status="Completed",
                                    recommendation=updated_recommendation,
                                    technical_score=technical_score,
                                    communication_score=communication_score,
                                    problem_solving_score=problem_solving_score,
                                    feedback=feedback,
                                    internal_notes=internal_notes,
                                )
                                st.success(
                                    "Interview scorecard saved permanently."
                                )
                                st.rerun()
                            except Exception as error:
                                st.error(
                                    f"Could not save scorecard: {error}"
                                )

                        refreshed = get_interview(
                            selected_interview_id
                        )

                        if refreshed:
                            overall = refreshed.get("overall_score")

                            metric1, metric2 = st.columns(2)

                            with metric1:
                                st.metric(
                                    "Overall Interview Score",
                                    (
                                        f"{float(overall):.2f}/5"
                                        if overall is not None
                                        else "Pending"
                                    ),
                                )

                            with metric2:
                                st.metric(
                                    "Recommendation",
                                    refreshed.get(
                                        "recommendation",
                                        "Pending",
                                    ),
                                )

                        # ------------------------------------------------
                        # Interview Audit History
                        # ------------------------------------------------
                        st.markdown("### 🕒 Interview Activity History")

                        events = get_interview_event_history(
                            selected_interview_id
                        )

                        if events:
                            event_rows = [
                                {
                                    "Event": item["event_type"],
                                    "Details": item["event_detail"],
                                    "Time (UTC)": item["created_at"],
                                }
                                for item in events
                            ]

                            st.dataframe(
                                pd.DataFrame(event_rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No interview activity recorded yet.")


    # ========================================================
    # RECRUITER COMMUNICATIONS TAB
    # ========================================================
    with communication_tab:
        st.subheader("📨 Recruiter Communications")
        st.caption(
            "Generate personalized recruiter messages from the candidate and interview data. "
            "Messages are saved to the local recruiter database for history."
        )

        if not st.session_state.job_fingerprint:
            st.info(
                "Analyze a Job Description and resumes first to activate recruiter communications."
            )
        else:
            communication_candidates = df.to_dict("records")

            candidate_options = {
                str(item.get("Resume", "")): str(
                    item.get("Candidate", "Candidate")
                )
                for item in communication_candidates
                if str(item.get("Resume", ""))
            }

            if not candidate_options:
                st.warning("No analyzed candidates are available.")
            else:
                selected_resume = st.selectbox(
                    "👤 Candidate",
                    list(candidate_options.keys()),
                    format_func=lambda value: (
                        f"{candidate_options[value]} — {value}"
                    ),
                    key="communication_candidate_selector",
                )

                selected_row = next(
                    (
                        item
                        for item in communication_candidates
                        if str(item.get("Resume", "")) == selected_resume
                    ),
                    None,
                )

                selected_cache = st.session_state.resume_cache.get(
                    selected_resume,
                    {}
                )

                selected_intelligence = selected_cache.get(
                    "intelligence",
                    {}
                )

                candidate_name = (
                    str(
                        selected_row.get(
                            "Candidate",
                            "Candidate",
                        )
                    )
                    if selected_row
                    else candidate_options[selected_resume]
                )

                candidate_email = str(
                    selected_intelligence.get(
                        "email",
                        ""
                    )
                    or ""
                )

                st.divider()

                # --------------------------------------------
                # Candidate contact
                # --------------------------------------------
                cc1, cc2, cc3 = st.columns(3)

                with cc1:
                    st.metric(
                        "Candidate",
                        candidate_name,
                    )

                with cc2:
                    st.write("**Email**")
                    st.write(
                        candidate_email
                        or "Not detected"
                    )

                with cc3:
                    if selected_intelligence.get("linkedin"):
                        st.link_button(
                            "LinkedIn",
                            selected_intelligence["linkedin"],
                        )
                    if selected_intelligence.get("github"):
                        st.link_button(
                            "GitHub",
                            selected_intelligence["github"],
                        )

                st.divider()

                # --------------------------------------------
                # Message settings
                # --------------------------------------------
                st.markdown("### ✉️ Compose Message")

                m1, m2 = st.columns(2)

                with m1:
                    template = st.selectbox(
                        "Message Template",
                        TEMPLATE_OPTIONS,
                        key="communication_template",
                    )

                    default_role = (
                        str(
                            st.session_state.jd_analysis.get(
                                "job_title",
                                ""
                            )
                            or st.session_state.jd_analysis.get(
                                "title",
                                ""
                            )
                            or st.session_state.jd_analysis.get(
                                "role",
                                ""
                            )
                            or "the position"
                        )
                    )

                    role_title = st.text_input(
                        "Job / Role Title",
                        value=default_role,
                        key="communication_role_title",
                    )

                    company_name = st.text_input(
                        "Company Name",
                        value="Our Company",
                        key="communication_company",
                    )

                    recruiter_name = st.text_input(
                        "Recruiter Name",
                        value="Recruitment Team",
                        key="communication_recruiter",
                    )

                with m2:
                    interview_rows = get_candidate_interview_history(
                        st.session_state.job_fingerprint,
                        selected_resume,
                    )

                    interview_map = {
                        str(item["id"]): item
                        for item in interview_rows
                    }

                    selected_interview = None

                    if interview_map:
                        selected_interview_id_for_message = st.selectbox(
                            "Interview Data (optional)",
                            ["None"] + list(interview_map.keys()),
                            format_func=lambda value: (
                                "None"
                                if value == "None"
                                else (
                                    f"#{value} — "
                                    f"{interview_map[value]['interview_round']} — "
                                    f"{interview_map[value]['scheduled_date']}"
                                )
                            ),
                            key="communication_interview_selector",
                        )

                        if selected_interview_id_for_message != "None":
                            selected_interview = interview_map[
                                selected_interview_id_for_message
                            ]

                    interview_context_key = (
                        str(selected_interview["id"])
                        if selected_interview
                        else "none"
                    )

                    interview_round = st.text_input(
                        "Interview Round",
                        value=(
                            selected_interview["interview_round"]
                            if selected_interview
                            else ""
                        ),
                        key=f"communication_round_{interview_context_key}",
                    )

                    interview_date = st.text_input(
                        "Interview Date",
                        value=(
                            selected_interview["scheduled_date"]
                            if selected_interview
                            else ""
                        ),
                        key=f"communication_date_{interview_context_key}",
                    )

                    interview_time = st.text_input(
                        "Interview Time",
                        value=(
                            selected_interview["scheduled_time"]
                            if selected_interview
                            else ""
                        ),
                        key=f"communication_time_{interview_context_key}",
                    )

                    interviewer = st.text_input(
                        "Interviewer",
                        value=(
                            selected_interview["interviewer"]
                            if selected_interview
                            else ""
                        ),
                        key=f"communication_interviewer_{interview_context_key}",
                    )

                    meeting_link = st.text_input(
                        "Meeting Link",
                        value=(
                            selected_interview["meeting_link"]
                            if selected_interview
                            else ""
                        ),
                        key=f"communication_meeting_link_{interview_context_key}",
                    )

                # --------------------------------------------
                # Feedback / scoring
                # --------------------------------------------
                feedback = st.text_area(
                    "Additional Feedback / Message Context",
                    value=(
                        selected_interview.get("feedback", "")
                        if selected_interview
                        else ""
                    ),
                    height=120,
                    placeholder=(
                        "Optional: add concise feedback, next-step details, "
                        "or context for the message."
                    ),
                    key="communication_feedback",
                )

                overall_score = (
                    selected_interview.get("overall_score", "")
                    if selected_interview
                    else ""
                )

                # --------------------------------------------
                # Generate
                # --------------------------------------------
                if st.button(
                    "✨ Generate Personalized Message",
                    type="primary",
                    use_container_width=True,
                    key="generate_communication_btn",
                ):
                    try:
                        generated = generate_communication(
                            template,
                            candidate_name=candidate_name,
                            candidate_email=candidate_email,
                            role_title=role_title,
                            company_name=company_name,
                            recruiter_name=recruiter_name,
                            interview_round=interview_round,
                            interview_date=interview_date,
                            interview_time=interview_time,
                            interviewer=interviewer,
                            meeting_link=meeting_link,
                            feedback=feedback,
                            overall_score=overall_score,
                        )

                        st.session_state.communication_generated = generated

                        st.success(
                            "Personalized recruiter message generated."
                        )
                    except Exception as error:
                        st.error(
                            f"Could not generate the message: {error}"
                        )

                generated = st.session_state.get(
                    "communication_generated"
                )

                if (
                    generated
                    and generated.get("template") == template
                ):
                    st.divider()
                    st.markdown("### 📨 Generated Email")

                    st.write("**To:**", generated.get("recipient") or "No email detected")
                    st.text_input(
                        "Subject",
                        value=generated.get("subject", ""),
                        key="generated_email_subject",
                    )

                    body_value = st.text_area(
                        "Email Body",
                        value=generated.get("body", ""),
                        height=300,
                        key="generated_email_body",
                    )

                    action1, action2, action3 = st.columns(3)

                    with action1:
                        mailto = build_mailto_url(
                            generated.get("recipient", ""),
                            generated.get("subject", ""),
                            body_value,
                        )

                        if generated.get("recipient"):
                            st.link_button(
                                "📧 Open in Email App",
                                mailto,
                                use_container_width=True,
                            )
                        else:
                            st.warning(
                                "No candidate email was detected, so an email-app link cannot be created."
                            )

                    with action2:
                        st.download_button(
                            "💾 Download .txt",
                            data=(
                                f"To: {generated.get('recipient', '')}\n"
                                f"Subject: {generated.get('subject', '')}\n\n"
                                f"{body_value}"
                            ),
                            file_name=(
                                f"{selected_resume.rsplit('.', 1)[0]}_"
                                f"{generated.get('communication_type', 'message')}.txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )

                    with action3:
                        if st.button(
                            "🗂️ Save to History",
                            use_container_width=True,
                            key="save_generated_communication",
                        ):
                            try:
                                communication_to_save = dict(generated)
                                communication_to_save["subject"] = (
                                    st.session_state.get(
                                        "generated_email_subject",
                                        generated.get("subject", ""),
                                    )
                                )
                                communication_to_save["body"] = body_value

                                save_communication(
                                    job_fingerprint=st.session_state.job_fingerprint,
                                    resume_name=selected_resume,
                                    candidate_name=candidate_name,
                                    candidate_email=candidate_email,
                                    communication=communication_to_save,
                                )

                                st.success(
                                    "Communication saved to recruiter history."
                                )
                                st.rerun()
                            except Exception as error:
                                st.error(
                                    f"Could not save communication: {error}"
                                )

                    st.caption(
                        "The email-app button opens a mailto draft; it does not send email automatically."
                    )

                # --------------------------------------------
                # Communication history
                # --------------------------------------------
                st.divider()
                st.markdown("### 🕒 Communication History")

                communication_history = list_communications(
                    st.session_state.job_fingerprint,
                    resume_name=selected_resume,
                )

                if not communication_history:
                    st.info(
                        "No saved recruiter communications for this candidate yet."
                    )
                else:
                    history_rows = [
                        {
                            "ID": item["id"],
                            "Type": item["communication_type"],
                            "Subject": item["subject"],
                            "Created (UTC)": item["created_at"],
                        }
                        for item in communication_history
                    ]

                    st.dataframe(
                        pd.DataFrame(history_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    history_id = st.selectbox(
                        "View Saved Message",
                        [item["id"] for item in communication_history],
                        key="communication_history_id",
                    )

                    selected_history_item = next(
                        (
                            item
                            for item in communication_history
                            if item["id"] == history_id
                        ),
                        None,
                    )

                    if selected_history_item:
                        st.write(
                            f"**Subject:** {selected_history_item['subject']}"
                        )
                        st.text_area(
                            "Saved Body",
                            value=selected_history_item["body"],
                            height=280,
                            key=f"saved_message_body_{history_id}",
                        )

    # ========================================================
    # RECRUITER WORKFLOW TAB
    # ========================================================
    with workflow_tab:
        st.subheader("🎯 Recruiter Workflow")

        summary = calculate_workflow_summary(
            df.to_dict("records"),
            workflow,
        )

        shortlist_rate = calculate_shortlist_rate(
            df.to_dict("records"),
            workflow,
        )

        w1, w2, w3, w4, w5 = st.columns(5)

        with w1:
            st.metric("Total", summary["total"])
        with w2:
            st.metric("New", summary["new"])
        with w3:
            st.metric("Reviewing", summary["reviewing"])
        with w4:
            st.metric("Shortlisted", summary["shortlisted"])
        with w5:
            st.metric("Shortlist Rate", f"{shortlist_rate:.1f}%")

        st.divider()

        f1, f2, f3 = st.columns(3)

        with f1:
            workflow_search = st.text_input(
                "🔎 Search",
                placeholder="Candidate or resume...",
                key="workflow_search",
            )

        with f2:
            workflow_status = st.selectbox(
                "Status",
                ["All"] + STATUS_OPTIONS,
                key="workflow_status_filter",
            )

        with f3:
            workflow_shortlisted = st.checkbox(
                "⭐ Shortlisted only",
                key="workflow_shortlisted_only",
            )

        f4, f5 = st.columns(2)

        with f4:
            workflow_min_ats = st.slider(
                "Minimum ATS Score",
                min_value=0,
                max_value=100,
                value=0,
                key="workflow_min_ats",
            )

        with f5:
            workflow_min_requirement = st.slider(
                "Minimum Requirement Match",
                min_value=0,
                max_value=100,
                value=0,
                key="workflow_min_requirement",
            )

        workflow_candidates = filter_candidates(
            candidates=df.to_dict("records"),
            workflow=workflow,
            search_text=workflow_search,
            status_filter=workflow_status,
            shortlisted_only=workflow_shortlisted,
            min_ats_score=workflow_min_ats,
            min_requirement_score=workflow_min_requirement,
        )

        st.write(
            f"Showing **{len(workflow_candidates)}** of **{len(df)}** candidates"
        )

        if workflow_candidates:
            workflow_rows = []

            for candidate in workflow_candidates:
                resume_name = str(candidate.get("Resume", ""))
                state = workflow.get(
                    resume_name,
                    {
                        "status": "New",
                        "shortlisted": False,
                        "notes": "",
                    },
                )

                workflow_rows.append(
                    {
                        "Candidate": candidate.get("Candidate", "Candidate"),
                        "Resume": resume_name,
                        "ATS Score": candidate.get("ATS Score", 0),
                        "Requirement Match": candidate.get("Requirement Match"),
                        "Status": state.get("status", "New"),
                        "Shortlisted": state.get("shortlisted", False),
                        "Recruiter Notes": state.get("notes", ""),
                    }
                )

            workflow_table = pd.DataFrame(workflow_rows)

            st.dataframe(
                workflow_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ATS Score": st.column_config.NumberColumn(
                        "ATS Score",
                        format="%.1f%%",
                    ),
                    "Requirement Match": st.column_config.NumberColumn(
                        "Requirement Match",
                        format="%.1f%%",
                    ),
                    "Shortlisted": st.column_config.CheckboxColumn(
                        "Shortlisted",
                    ),
                },
            )

            st.divider()

            workflow_resume_options = [
                str(item.get("Resume", ""))
                for item in workflow_candidates
            ]

            selected_workflow_resume = st.selectbox(
                "Manage Candidate",
                workflow_resume_options,
                key="workflow_candidate_selector",
            )

            current_state = workflow.get(
                selected_workflow_resume,
                {
                    "status": "New",
                    "shortlisted": False,
                    "notes": "",
                },
            )

            current_candidate_name = next(
                (
                    str(item.get("Candidate", "Candidate"))
                    for item in workflow_candidates
                    if str(item.get("Resume", "")) == selected_workflow_resume
                ),
                "Candidate",
            )

            st.markdown(f"### 👤 {current_candidate_name}")

            manage1, manage2 = st.columns(2)

            with manage1:
                manage_status = st.selectbox(
                    "Status",
                    STATUS_OPTIONS,
                    index=(
                        STATUS_OPTIONS.index(
                            current_state.get("status", "New")
                        )
                        if current_state.get("status", "New") in STATUS_OPTIONS
                        else 0
                    ),
                    key=f"workflow_manage_status_{safe_filename(selected_workflow_resume)}",
                )

            with manage2:
                if current_state.get("shortlisted", False):
                    manage_button_text = "⭐ Remove from Shortlist"
                else:
                    manage_button_text = "⭐ Add to Shortlist"

                if st.button(
                    manage_button_text,
                    key=f"workflow_manage_shortlist_{safe_filename(selected_workflow_resume)}",
                    use_container_width=True,
                ):
                    set_shortlist(
                        workflow,
                        selected_workflow_resume,
                        not current_state.get("shortlisted", False),
                    )
                    persist_workflow_state(
                        selected_workflow_resume,
                        workflow[selected_workflow_resume],
                    )
                    st.session_state.candidate_workflow = workflow
                    st.rerun()

            if manage_status != current_state.get("status", "New"):
                set_candidate_status(
                    workflow,
                    selected_workflow_resume,
                    manage_status,
                )
                persist_workflow_state(
                    selected_workflow_resume,
                    workflow[selected_workflow_resume],
                )
                st.session_state.candidate_workflow = workflow
                st.rerun()

            manage_notes = st.text_area(
                "📝 Recruiter Notes",
                value=current_state.get("notes", ""),
                height=140,
                placeholder="Add interview/screening notes...",
                key=f"workflow_manage_notes_{safe_filename(selected_workflow_resume)}",
            )

            if st.button(
                "💾 Save Candidate Notes",
                key=f"workflow_manage_save_{safe_filename(selected_workflow_resume)}",
                use_container_width=True,
            ):
                set_candidate_notes(
                    workflow,
                    selected_workflow_resume,
                    manage_notes,
                )
                persist_workflow_state(
                    selected_workflow_resume,
                    workflow[selected_workflow_resume],
                )
                st.session_state.candidate_workflow = workflow
                st.success("Notes saved permanently.")
                st.rerun()

        else:
            st.warning("No candidates match the workflow filters.")

    # ========================================================
    # EXPORTS
    # ========================================================
    st.divider()
    st.header("📥 Export Reports")

    export_df = df.copy()
    export_df["Status"] = export_df["Resume"].apply(
        lambda x: workflow.get(str(x), {}).get("status", "New")
    )
    export_df["Shortlisted"] = export_df["Resume"].apply(
        lambda x: bool(workflow.get(str(x), {}).get("shortlisted", False))
    )
    export_df["Recruiter Notes"] = export_df["Resume"].apply(
        lambda x: workflow.get(str(x), {}).get("notes", "")
    )

    export1, export2 = st.columns(2)

    with export1:
        excel_file = create_excel_report(export_df)

        st.download_button(
            "📊 Download Excel",
            data=excel_file,
            file_name="candidate_screening_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    with export2:
        if st.session_state.pdf_reports:
            zip_file = create_zip_reports(
                st.session_state.pdf_reports
            )

            st.download_button(
                "📦 Download PDF Reports",
                data=zip_file,
                file_name="candidate_reports.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.info("No PDF reports available.")

else:
    # ========================================================
    # LANDING PAGE
    # ========================================================
    st.divider()
    st.header("🚀 Platform Capabilities")

    capabilities = [
        (
            "📄 Document Intelligence",
            "Extract text and hidden PDF/DOCX hyperlink targets."
        ),
        (
            "👤 Candidate Intelligence",
            "Extract name, email, phone, LinkedIn, GitHub, education and projects."
        ),
        (
            "🧠 Semantic Matching",
            "Batch Sentence Transformer semantic analysis."
        ),
        (
            "🛠️ Skill Intelligence",
            "Detect technical skills and categorize them."
        ),
        (
            "🎯 Requirement Matching",
            "Separate required and preferred skills from other requirements."
        ),
        (
            "📄 Resume Quality",
            "Analyze structure, completeness, repetition and measurable content."
        ),
        (
            "🏆 Candidate Ranking",
            "Compare multiple resumes against the same Job Description."
        ),
        (
            "🎯 Recruiter Workflow",
            "Track status, shortlist candidates and save recruiter notes."
        ),
        (
            "🗓️ Interview Management",
            "Schedule interview rounds, manage meeting details, record scorecards and preserve interview history."
        ),
        (
            "📨 Recruiter Communications",
            "Generate personalized interview, follow-up, selection, hold and rejection messages with saved history."
        ),
        (
            "🧠 AI-Assisted Screening Brief",
            "Convert ATS, requirement, skill and quality signals into recruiter-ready strengths, gaps and interview focus areas."
        ),
        (
            "✨ Free Gemini-Enhanced Screening",
            "Use Google's free-tier Gemini API to synthesize existing candidate evidence into recruiter-ready insights."
        ),
        (
            "👤 Candidate 360°",
            "View contact data, scores, requirements, skills, semantic analysis and quality in one profile."
        ),
        (
            "📥 Recruiter Reports",
            "Export screening results to Excel and PDF."
        ),
        (
            "📋 JD Requirement Matrix",
            "Convert Job Description intelligence into structured requirements "
            "with priority, source, evidence and candidate coverage."
        ),
        (
            "💾 Persistent Recruiter Data",
            "Store candidate workflow state in a local SQLite database."
        ),
        (
            "🕒 Pipeline History",
            "Track status transitions and recruiter note updates over time."
        ),
    ]

    for title, description in capabilities:
        st.markdown(f"### {title}")
        st.write(description)

    st.divider()
    st.header("🏗️ Architecture")

    st.code(
        """
Job Description + Resumes
             │
             ▼
      Document Extraction
       ┌─────┴─────┐
       ▼           ▼
 Resume Info     JD Info
       │           │
       └─────┬─────┘
             ▼
      Batch AI Analysis
             │
      ┌──────┼────────┐
      ▼      ▼        ▼
    Skills Semantic  Lexical
      │      │        │
      └──────┼────────┘
             ▼
        Hybrid ATS
             │
      ┌──────┴──────────┐
      ▼                 ▼
 Requirements      Resume Quality
      │                 │
      └────────┬────────┘
               ▼
       Candidate 360° View
               │
               ▼
       Recruiter Workflow
        ┌──────┼──────┐
        ▼      ▼      ▼
      Status Shortlist Notes
               │
               ▼
       SQLite Persistence
        ┌───────────────┐
        ▼               ▼
 Status History   Notes History
        │               │
        └──────┬────────┘
               ▼
        Excel / PDF Reports
        """,
        language="text",
    )
