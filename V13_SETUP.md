# V13 Setup Checklist

1. Stop the running API with Ctrl+C.
2. Copy the V13 backend files over the V11/V12 backend files in the existing CV Analyzer folder.
3. Keep the existing AI analysis modules from V9/V10.
4. Install `requirements-api-v13.txt`.
5. Set a 32+ character `JWT_SECRET`.
6. Start FastAPI on port 8000.
7. In `frontend/`, run `npm install` and `npm run dev`.
8. Open http://127.0.0.1:5173.

Expected backend checks:

```text
GET /docs -> 200 OK
GET /openapi.json -> 200 OK
```

Expected workflow:

Register → Login → Create Job → Paste JD → Upload resumes → View candidates → Open Candidate 360° → change workflow status.
