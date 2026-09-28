const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

export function getApiBase() {
  return API_BASE
}

function authHeaders() {
  const token = localStorage.getItem('cv_access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(path, options = {}) {
  const headers = {
    ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
    ...authHeaders(),
    ...(options.headers || {})
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers })
  let payload = null
  try { payload = await response.json() } catch { payload = null }
  if (!response.ok) {
    const detail = payload?.detail || `Request failed (${response.status})`
    throw new Error(detail)
  }
  return payload
}

export const api = {
  register: (body) => request('/api/v1/auth/register', { method: 'POST', body: JSON.stringify(body) }),
  login: (body) => request('/api/v1/auth/login', { method: 'POST', body: JSON.stringify(body) }),
  me: () => request('/api/v1/auth/me'),
  jobs: () => request('/api/v1/jobs'),
  createJob: (body) => request('/api/v1/jobs', { method: 'POST', body: JSON.stringify(body) }),
  updateJob: (id, body) => request(`/api/v1/jobs/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteJob: (id) => request(`/api/v1/jobs/${id}`, { method: 'DELETE' }),
  candidates: (jobId) => request(`/api/v1/jobs/${jobId}/candidates`),
  candidate: (jobId, candidateId) => request(`/api/v1/jobs/${jobId}/candidates/${candidateId}`),
  resumeVersions: (candidateId) => request(`/api/v1/candidates/${candidateId}/resume-versions`),
  compareResumeVersions: (candidateId, version1, version2, jobId = null) => {
    const params = new URLSearchParams({
      version1: String(version1),
      version2: String(version2)
    })

    if (jobId) {
      params.set('job_id', jobId)
    }

    return request(
      `/api/v1/candidates/${candidateId}/resume-compare?${params.toString()}`
    )
  },
  updateCandidateStatus: (jobId, candidateId, status) => request(`/api/v1/jobs/${jobId}/candidates/${candidateId}/status`, {
    method: 'PATCH', body: JSON.stringify({ status })
  }),
  auditLogs: (limit = 100) => request(`/api/v1/audit-logs?limit=${limit}`),
  evaluationRuns: (jobId, limit = 20) => request(`/api/v1/jobs/${jobId}/evaluation-runs?limit=${limit}`),
  evaluateJob: (jobId, body) => request(`/api/v1/jobs/${jobId}/evaluate`, { method: 'POST', body: JSON.stringify(body) }),
  analyzeJob: async (jobId, files, topN = 20) => {
    const form = new FormData()
    files.forEach((file) => form.append('resumes', file))
    return request(`/api/v1/jobs/${jobId}/analyze?top_n_keywords=${topN}&include_raw_text=false`, {
      method: 'POST', body: form
    })
  }
}
