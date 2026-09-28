
# CV Analyzer — AI Resume Screening & Job Matching Platform

An end-to-end AI recruitment platform for screening resumes against job descriptions, explaining ATS signals, managing candidates through a recruiter workflow, and preserving resume and application history.

## Project status

This project is currently verified at the V17.7 checkpoint.

Verified local services:
- Frontend: http://localhost:8081
- API: http://localhost:8000
- API health: http://localhost:8000/health
- Database: PostgreSQL 17

## Highlights

- Recruiter and job management
- Batch resume upload and screening
- Hybrid ATS scoring using skill, semantic, lexical, and keyword signals
- Job requirement extraction and requirement coverage analysis
- Explainable screening with matched and missing skills
- Candidate 360° recruiter workflow
- Candidate status tracking and application history
- Interview management and recruiter communication tools
- Recruiter analytics and audit logging
- RAG-powered recruiter copilot support
- Candidate comparison and duplicate detection
- Resume version history tracking (v1, v2, v3...)
- Exact application-to-resume-version linkage
- Job-specific analysis snapshots
- Resume comparison UI for ATS and skill differences
- Dockerized deployment with PostgreSQL
- FastAPI backend + React/Vite frontend

## Resume identity and versioning

The platform distinguishes between three upload cases:

1. Same file
   - SHA-256 hash match
   - Reuse the existing resume/application

2. New file for an existing candidate
   - Create the next Resume Version
   - Previous versions remain historical

3. New candidate
   - Create Candidate + Resume v1

Applications are linked to the exact resume version used to screen the candidate for that job.

This is the current behavior verified in V17.7:
- Resume v1 → v2 creation works
- Historical v1 reuse works
- Application → exact resume version binding works
- Duplicate detection and re-upload messaging works

## Architecture overview

```text
Job Description + Resume(s)
        ↓
Document Extraction
        ↓
Resume / JD Intelligence
        ↓
┌─────────────────────────────┐
│ Skill Matching              │
│ Sentence-Transformers      │
│ Lexical / TF-IDF            │
│ Keyword Matching            │
└─────────────────────────────┘
        ↓
   Hybrid ATS Engine
        ↓
Requirement Matching
        ↓
Explainability + Evidence
        ↓
Recruiter Candidate Pipeline
        ↓
Candidate 360° / Analytics / Audit
```

## Tech stack

### Backend
- Python 3.11
- FastAPI
- SQLAlchemy 2.x
- Alembic
- PostgreSQL 17
- Pydantic
- JWT authentication
- scikit-learn
- Sentence Transformers
- NLTK
- Gemini integration
- Hybrid semantic + lexical matching
- Explainability and evidence extraction
- RAG recruiter copilot

### Frontend
- React
- Vite
- JavaScript / JSX
- CSS

### DevOps
- Docker
- Docker Compose
- Nginx
- GitHub Actions
- GitHub Container Registry
- Render deployment configuration

## Repository structure

```text
cv-analyzer/
├── backend/
├── frontend/
├── docs/
├── scripts/
├── app.py
├── analysis.py
├── ats_engine.py
├── candidate_comparison.py
├── candidate_profile.py
├── database.py
├── docker-compose.yml
├── Dockerfile.api
├── interview_manager.py
├── jd_intelligence.py
├── matching_engine.py
├── models.py
├── README.md
├── recruiter_analytics.py
├── recruiter_kanban.py
├── recruiter_store.py
├── recruiter_workflow.py
├── resume_intelligence.py
├── resume_quality.py
├── resume_utils.py
├── semantic_matcher.py
├── screening_brief.py
├── security.py
├── responsible_ai.py
└── requirements.txt
```

## Security and responsible AI

- Recruiter data is scoped by the authenticated owner
- Passwords are stored as hashes, not plaintext
- JWT bearer authentication protects API access
- Duplicate detection uses SHA-256 file hashes
- Audit logging records important platform actions
- Resume content is not intentionally embedded into audit metadata
- The platform is designed as decision-support software, not automatic hiring enforcement

## Local setup

### Prerequisites
- Docker Desktop
- Git
- Node.js / npm
- Python 3.11

### Run locally

```bash
git clone https://github.com/AdhyanSharma/cv-analyzer.git
cd cv-analyzer
docker compose up -d --build
```

Then open:
- Frontend: http://localhost:8081
- API health: http://localhost:8000/health

### Frontend development build

```bash
cd frontend
npm install
npm run build
```

### Backend compile check

```bash
python -m py_compile backend/models.py
python -m py_compile backend/platform_store.py
python -m py_compile backend/v11_schemas.py
python -m py_compile backend/api.py
```

## API areas

Examples of the current platform API surface:

```text
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me

GET  /api/v1/jobs
POST /api/v1/jobs
PATCH /api/v1/jobs/{job_id}
DELETE /api/v1/jobs/{job_id}

POST /api/v1/jobs/{job_id}/analyze
GET  /api/v1/jobs/{job_id}/candidates
GET  /api/v1/jobs/{job_id}/candidates/{candidate_id}
PATCH /api/v1/jobs/{job_id}/candidates/{candidate_id}/status

GET /api/v1/candidates/{candidate_id}/resume-versions
GET /api/v1/candidates/{candidate_id}/resume-compare

GET /api/v1/audit-logs
GET /api/v1/jobs/{job_id}/evaluation-runs
POST /api/v1/jobs/{job_id}/evaluate
GET /api/v1/responsible-ai/notice
```

## Deployment

The repository includes Docker and Render configuration for cloud deployment.

Use the live URL from your Render dashboard for the public site. The exact public onrender.com link is deployment-specific and should be confirmed in the deployed project dashboard.

## GitHub and project links

- GitHub: https://github.com/AdhyanSharma/cv-analyzer
- Local frontend: http://localhost:8081
- Local API: http://localhost:8000/health

## V17.7 verification summary

The current project checkpoint was validated for:
- Resume v1 → v2 creation
- Historical v1 reuse
- Candidate identity preservation
- Application → exact resume-version binding
- Job-specific analysis snapshots
- PostgreSQL and Docker health
- React build integrity

### Version track
- V17.3 Resume Versioning
- V17.4 Application → Resume Version
- V17.5 Job-specific Analysis Snapshots
- V17.5.1 Resume Comparison UI
- V17.6 Re-upload / Version Messaging
- V17.7 Final Regression + Docker Verification

Updates remaining: 0

## Author

Adhyan Sharma

AI Undergraduate | Python | Machine Learning | NLP | Deep Learning | FastAPI | React | PostgreSQL


📁 Major Project Modules
app.py
analysis.py
semantic_matcher.py
ats_engine.py
resume_utils.py
resume_intelligence.py
resume_quality.py
jd_intelligence.py
matching_engine.py
recruiter_workflow.py
recruiter_store.py
recruiter_kanban.py
candidate_profile.py
interview_manager.py
recruiter_communication.py
screening_brief.py
gemini_screening_brief.py
rag_recruiter_copilot.py
candidate_comparison.py
jd_intelligence_v2.py
jd_intelligence_v3.py
jd_intelligence_v4.py
jd_intelligence_v5.py
jd_intelligence_v6.py
jd_requirement_matrix.py
recruiter_analytics.py
explainability_engine.py
backend/
frontend/
🔐 Data & Security Notes
Recruiter data is scoped by authenticated owner.

Passwords are stored as hashes, not plaintext.

JWT bearer authentication protects platform API access.

Resume duplicate detection uses SHA-256 file hashes.

Audit logging records important recruiter/platform actions.

Resume content is not intentionally written into audit metadata.

API and database run as separate Docker services.

⚖️ Responsible AI
The screening engine is designed as decision-support software. ATS scores and detected requirements are signals for recruiter review rather than automatic hiring decisions. Missing evidence should not be treated as proof that a candidate lacks a skill.

🧪 Verification Status
The current V17.7 local checkpoint was regression-tested for:

Resume v1 → v2 creation

Historical v1 reuse

Candidate identity preservation

Application → exact resume-version binding

Job-specific resume analysis snapshots

PostgreSQL migrations through 0006_resume_version_analysis

Backend Python compilation

React production build

Docker Compose services

API health

Current local services
Service	Local URL
React frontend	http://localhost:8081
FastAPI backend	http://localhost:8000
PostgreSQL	5432 (Docker internal/exposed service)
▶️ Run Locally
Prerequisites
Docker Desktop

Git

Node.js / npm (for direct frontend development)

Python 3.11 (for direct backend development)

Docker
git clone https://github.com/AdhyanSharma/cv-analyzer.git
cd cv-analyzer
docker compose up -d --build
Frontend:

http://localhost:8081
API health:

http://localhost:8000/health
Frontend development build
cd frontend
npm install
npm run build
Backend compilation check
python -m py_compile backend/models.py
python -m py_compile backend/platform_store.py
python -m py_compile backend/v11_schemas.py
python -m py_compile backend/api.py
🔑 Environment Variables
Typical deployment variables include:

DATABASE_URL=
JWT_SECRET=
GEMINI_API_KEY=
CORS_ORIGINS=
UVICORN_WORKERS=2
Never commit real secrets or .env files.

🌐 API Areas
Examples of the current API surface:

POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me

GET  /api/v1/jobs
POST /api/v1/jobs
PATCH /api/v1/jobs/{job_id}
DELETE /api/v1/jobs/{job_id}

POST /api/v1/jobs/{job_id}/analyze
GET  /api/v1/jobs/{job_id}/candidates
GET  /api/v1/jobs/{job_id}/candidates/{candidate_id}
PATCH /api/v1/jobs/{job_id}/candidates/{candidate_id}/status

GET /api/v1/candidates/{candidate_id}/resume-versions
GET /api/v1/candidates/{candidate_id}/resume-compare

GET /api/v1/audit-logs
GET /api/v1/jobs/{job_id}/evaluation-runs
POST /api/v1/jobs/{job_id}/evaluate
GET /api/v1/responsible-ai/notice
📊 ATS Scoring
The project currently uses a hybrid project-defined weighting approach:

Skill Match     35%
Semantic Match  30%
Lexical Match   20%
Keyword Match   15%
-------------------
Total           100%
These weights are implementation choices for this project, not an industry-standard ATS formula.

☁️ Deployment
The repository contains Docker and Render configuration for cloud deployment.

For the public demo, use the Render frontend service URL configured in your Render dashboard. The exact public onrender.com URL is deployment-specific and is intentionally not hard-coded here until the live service URL is verified.

Render supports Docker-based web services and managed PostgreSQL deployments. See the official Render documentation for current deployment behavior: https://render.com/docs

🔗 Project Links
GitHub: https://github.com/AdhyanSharma/cv-analyzer

Local Demo: http://localhost:8081

Local API: http://localhost:8000/health

📌 Portfolio Description
CV Analyzer is a full-stack AI recruitment platform that combines hybrid ATS scoring, semantic resume matching, job requirement intelligence, explainable screening, recruiter workflow management, candidate analytics, auditability, and resume version tracking. It is built with FastAPI, React, PostgreSQL, SQLAlchemy, Docker, and modern NLP/LLM components.

👤 Author
Adhyan Sharma

AI Undergraduate | Python | Machine Learning | NLP | Deep Learning | FastAPI | React | PostgreSQL

