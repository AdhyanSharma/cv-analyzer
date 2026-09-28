

c2df6dc7-ec2d-4a6c-93b1-c0593d2e1ac7.png


Pasted text(6).txt
Document
and also that 2 resume i uploaded was same 


Pasted text(7).txt
Document
and also tell me how many upgrades are left 


Pasted text(8).txt
Document


Pasted text(9).txt
Document


Pasted text(10).txt
Document


Pasted text(20260928-051442).txt
Document


Pasted text(20260928-055653).txt
Document


Pasted text(20260928-061343).txt
Document

🎉 DONE — V17.7 is fully green!

Your final verification shows:

Frontend build                 ✅
frontend/dist/index.html       ✅
API container                  ✅ Healthy
PostgreSQL                     ✅ Healthy
React frontend                 ✅ Running
Resume v1 → v2                 ✅
Historical v1 reuse            ✅
Application → exact version   ✅
🏁 Resume Versioning Track
✅ V17.3 Resume Versioning
✅ V17.4 Application → Resume Version
✅ V17.5 Job-specific Analysis Snapshots
✅ V17.5.1 Resume Comparison UI
✅ V17.6 Re-upload / Version Messaging
✅ V17.7 Final Regression + Docker Verification
Updates remaining: 0.

Your local platform is currently running at:

Frontend → http://localhost:8081
API      → http://localhost:8000
Database → PostgreSQL 17
One final housekeeping step
Before moving on to the next product upgrade, save this stable state in Git:

cd "C:\Users\ASUS\CV Analyzer"

git status
git add backend frontend docker-compose.yml
git commit -m "feat: add resume versioning and comparison"
git push
🚀 This is now a solid V17.7 checkpoint.

The next work can move beyond resume versioning into the next major recruiter-platform capability rather than modifying this foundation.


done also provide me the readme and site link

🎉 Done! Your updated README is ready and covers

🎉 Done! I prepared the README with the current V17.7 architecture, features, setup, API areas, resume versioning, security, responsible-AI notes, and verified local service URLs.

README
Download the updated README.md

Your currently verified local site
Frontend: http://localhost:8081

API health: http://localhost:8000/health

GitHub repository: Adhyan Sharma / cv-analyzer

For the public Render site, I don't have a verified live onrender.com URL in the available evidence, so I haven't invented one. The README leaves that deployment-specific URL for you to fill from the Render dashboard. Render's current documentation confirms that deployed services receive public URLs and can be configured through its deployment setup. 

Your project checkpoint is now V17.7 ✅


README.md
Document


how to commit it into my git

Absolutely! 🚀 Since your project is already connected to GitHub, adding the README is very simple.

1. Download the README
Download README.md

Save it here:

C:\Users\ASUS\CV Analyzer\README.md
Replace the existing README.md if one is already there.

2. Open PowerShell in your project
cd "C:\Users\ASUS\CV Analyzer"
3. Check what changed
git status
You should see something like:

modified: README.md
or:

README.md
4. Add the README
git add README.md
5. Commit it
git commit -m "docs: update README"
6. Push to GitHub
git push
That's it. 🎉

Verify
Run:

git status
You should ideally get:

nothing to commit, working tree clean
Then open your repository:

https://github.com/AdhyanSharma/cv-analyzer

GitHub will automatically display the README.md on the repository homepage.

🚀 One-command version
After saving the README in your project folder:

cd "C:\Users\ASUS\CV Analyzer"; git add README.md; git commit -m "docs: update README"; git push
Since you already committed the V17.7 feature work, this will create a clean separate documentation commit.


README.md
Document


Library
/
README.md



CV Analyzer — AI Resume Screening & Job Matching Platform
An end-to-end AI recruitment platform for screening resumes against job descriptions, explaining ATS signals, managing candidates through a recruiter workflow, and preserving resume/application history.

🚀 Highlights
Job and recruiter account management

Multi-resume upload and batch screening

Hybrid ATS scoring using skill, semantic, lexical, and keyword signals

Job requirement extraction and requirement coverage

Explainable screening with matched/missing skills and resume evidence

Candidate 360° recruiter view

Candidate status workflow and application tracking

Interview management and recruiter communications

Recruiter analytics, evaluation telemetry, and audit logs

RAG-based recruiter copilot support

Candidate comparison

Exact duplicate-resume detection with SHA-256 hashing

Candidate resume version history (v1, v2, v3...)

Application binding to the exact resume version used for screening

Job-specific analysis snapshots for every resume version

Resume version comparison with ATS/skill changes

Dockerized deployment with PostgreSQL

FastAPI backend + React/Vite frontend

CI/CD with GitHub Actions and container publishing

🧠 Screening Architecture
Job Description + Resume(s)
            ↓
     Document Extraction
            ↓
 Resume / JD Intelligence
            ↓
 ┌─────────────────────────────┐
 │ Skill Matching              │
 │ Sentence-Transformer        │
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
📄 Resume Identity & Versioning
The platform distinguishes three upload cases:

Same file
   ↓
SHA-256 hash match
   ↓
Reuse existing resume/application

New file for existing candidate
   ↓
Create next Resume Version
   ↓
Previous version remains historical

New candidate
   ↓
Create Candidate + Resume v1
Applications are linked to the exact resume version that produced the screening analysis.

🏗️ Tech Stack
Backend
Python 3.11

FastAPI

SQLAlchemy 2.x

Alembic

PostgreSQL 17

Pydantic

JWT authentication

AI / NLP
scikit-learn

Sentence Transformers

NLTK

Gemini integration

Hybrid semantic + lexical matching

Explainability and evidence extraction

RAG recruiter copilot

Frontend
React

Vite

JavaScript / JSX

CSS

DevOps
Docker

Docker Compose

Nginx

GitHub Actions

GitHub Container Registry

Render deployment configuration

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

