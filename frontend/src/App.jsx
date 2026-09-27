import React, { useEffect, useMemo, useState } from 'react'
import { api, getApiBase } from './api'

const STATUS_OPTIONS = ['new', 'reviewing', 'shortlisted', 'hold', 'rejected']

function scoreOf(analysis, key) {
  const candidates = [
    analysis?.[key],
    analysis?.scores?.[key],
    analysis?.score_breakdown?.[key],
    analysis?.explainability?.score_breakdown?.[key],
  ]
  for (const value of candidates) {
    if (typeof value === 'number') return value
  }
  return null
}

function atsOf(analysis) {
  const direct = analysis?.ats_score ?? analysis?.ats?.score ?? analysis?.explainability?.ats_score
  return typeof direct === 'number' ? direct : scoreOf(analysis, 'final_score')
}

function arrOf(analysis, key, fallback = []) {
  const places = [
    analysis?.[key],
    analysis?.requirements?.[key],
    analysis?.explainability?.[key],
    analysis?.explainability?.[`${key}_skills`]
  ]
  for (const value of places) if (Array.isArray(value)) return value
  return fallback
}

function textOf(value) {
  if (value == null) return ''
  if (typeof value === 'string') return value
  return String(value)
}

function scoreClass(value) {
  if (value == null) return 'muted'
  if (value >= 80) return 'good'
  if (value >= 60) return 'mid'
  return 'low'
}

function App() {
  const [user, setUser] = useState(null)
  const [loadingAuth, setLoadingAuth] = useState(true)

  useEffect(() => {
    if (!localStorage.getItem('cv_access_token')) return setLoadingAuth(false)
    api.me().then((r) => setUser(r.user)).catch(() => localStorage.removeItem('cv_access_token')).finally(() => setLoadingAuth(false))
  }, [])

  if (loadingAuth) return <LoadingScreen />
  if (!user) return <AuthScreen onAuthenticated={setUser} />
  return <Dashboard user={user} onLogout={() => { localStorage.removeItem('cv_access_token'); setUser(null) }} />
}

function LoadingScreen() {
  return <div className="screen-center"><div className="loader" /><div>Loading recruiter workspace…</div></div>
}

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState({ email: '', full_name: '', password: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event) {
    event.preventDefault(); setBusy(true); setError('')
    try {
      const result = mode === 'login' ? await api.login({ email: form.email, password: form.password }) : await api.register(form)
      localStorage.setItem('cv_access_token', result.access_token)
      onAuthenticated(result.user)
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  return <div className="auth-shell">
    <div className="auth-panel brand-panel">
      <div className="logo-mark">CV</div>
      <div className="eyebrow">AI RECRUITER PLATFORM · V13</div>
      <h1>Recruiter workspace, built around your screening pipeline.</h1>
      <p>Manage jobs, analyze resumes, inspect evidence, and move candidates through a job-specific pipeline.</p>
      <div className="feature-row"><span>⚡ FastAPI</span><span>🧠 ATS + Semantic</span><span>🔍 Explainable</span></div>
    </div>
    <div className="auth-panel form-panel">
      <div className="tabs">
        <button className={mode === 'login' ? 'tab active' : 'tab'} onClick={() => setMode('login')}>Sign in</button>
        <button className={mode === 'register' ? 'tab active' : 'tab'} onClick={() => setMode('register')}>Create account</button>
      </div>
      <form onSubmit={submit} className="stack gap-lg">
        <div><h2>{mode === 'login' ? 'Welcome back' : 'Create recruiter account'}</h2><p className="muted">{mode === 'login' ? 'Sign in to your recruitment workspace.' : 'Your account keeps jobs and candidates scoped to you.'}</p></div>
        {mode === 'register' && <Field label="Full name"><input value={form.full_name} onChange={(e) => setForm({...form, full_name:e.target.value})} required placeholder="Adhyan Sharma" /></Field>}
        <Field label="Email"><input type="email" value={form.email} onChange={(e) => setForm({...form, email:e.target.value})} required placeholder="recruiter@company.com" /></Field>
        <Field label="Password"><input type="password" value={form.password} onChange={(e) => setForm({...form, password:e.target.value})} required minLength={8} placeholder="••••••••" /></Field>
        {error && <div className="error-box">{error}</div>}
        <button className="primary btn-lg" disabled={busy}>{busy ? 'Working…' : mode === 'login' ? 'Sign in →' : 'Create account →'}</button>
      </form>
    </div>
  </div>
}

function Field({ label, children }) { return <label className="field"><span>{label}</span>{children}</label> }

function Dashboard({ user, onLogout }) {
  const [jobs, setJobs] = useState([])
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [candidateRows, setCandidateRows] = useState([])
  const [selectedCandidate, setSelectedCandidate] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [activeView, setActiveView] = useState('overview')
  const [creating, setCreating] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const selectedJob = jobs.find((job) => job.id === selectedJobId) || null
  const stats = useMemo(() => ({
    jobs: jobs.length,
    open: jobs.filter((j) => j.status === 'open').length,
    candidates: candidateRows.length,
    shortlisted: candidateRows.filter((c) => c.status === 'shortlisted').length
  }), [jobs, candidateRows])

  async function loadJobs(preselect = null) {
    try {
      const rows = await api.jobs(); setJobs(rows)
      const id = preselect || selectedJobId || rows[0]?.id || null
      setSelectedJobId(id)
    } catch (err) { setError(err.message) }
  }

  async function loadCandidates(jobId) {
    if (!jobId) return setCandidateRows([])
    try { setCandidateRows(await api.candidates(jobId)) } catch (err) { setError(err.message) }
  }

  useEffect(() => { loadJobs() }, [])
  useEffect(() => { loadCandidates(selectedJobId); setSelectedCandidate(null) }, [selectedJobId])

  async function createJob(form) {
    setCreating(true); setError('')
    try { const job = await api.createJob(form); await loadJobs(job.id); setShowCreate(false); setMessage('Job created.') } catch(err) { setError(err.message) } finally { setCreating(false) }
  }

  async function uploadResumes(files) {
    if (!selectedJob) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await api.analyzeJob(selectedJob.id, files)
      setCandidateRows((prev) => mergeCandidates(prev, result))
      setMessage(`${result.length} candidate application(s) analyzed.`)
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  async function updateStatus(candidateId, status) {
    try {
      const row = await api.updateCandidateStatus(selectedJob.id, candidateId, status)
      setCandidateRows((prev) => prev.map((x) => x.candidate_id === candidateId ? row : x))
      if (selectedCandidate?.candidate_id === candidateId) setSelectedCandidate(row)
    } catch(err) { setError(err.message) }
  }

  async function deleteSelectedJob() {
    if (!selectedJob || !confirm(`Delete ${selectedJob.title}?`)) return
    try { await api.deleteJob(selectedJob.id); setSelectedJobId(null); await loadJobs(); setMessage('Job deleted.') } catch(err) { setError(err.message) }
  }

  return <div className="app-shell">
    <header className="topbar">
      <div className="topbar-left"><div className="logo-mark small">CV</div><div><div className="top-title">CV Analyzer</div><div className="top-sub">Recruiter Command Center</div></div></div>
      <div className="topbar-right"><span className="api-dot">● API connected</span><span className="user-chip">{user.full_name}</span><button className="ghost" onClick={onLogout}>Log out</button></div>
    </header>
    <div className="body-grid">
      <aside className="sidebar">
        <div className="side-section"><div className="side-label">WORKSPACE</div><button className={activeView==='overview'?'side-link active':'side-link'} onClick={() => setActiveView('overview')}>▦ Overview</button><button className={activeView==='insights'?'side-link active':'side-link'} onClick={() => setActiveView('insights')}>◫ Evaluation & Audit</button><button className="side-link" onClick={() => { setActiveView('overview'); setShowCreate(true) }}>＋ New job</button></div>
        <div className="side-section"><div className="side-label">JOBS</div>{jobs.map(job => <button key={job.id} className={job.id === selectedJobId ? 'job-link selected' : 'job-link'} onClick={() => setSelectedJobId(job.id)}><span className={`status-dot ${job.status}`} /><span className="truncate">{job.title}</span></button>)}{jobs.length===0 && <div className="muted small-text">No jobs yet.</div>}</div>
      </aside>
      <main className="content">
        {activeView === 'overview' ? <>
          <div className="page-head"><div><div className="eyebrow">RECRUITER DASHBOARD · V14</div><h1>{selectedJob ? selectedJob.title : 'Your workspace'}</h1><p>{selectedJob ? `${selectedJob.company || 'Company not specified'} · ${selectedJob.status}` : 'Create a job to start screening candidates.'}</p></div><div className="head-actions"><button className="secondary" onClick={() => loadJobs()}>↻ Refresh</button><button className="primary" onClick={() => setShowCreate(true)}>＋ New job</button></div></div>
          {error && <div className="error-box global">{error}<button onClick={() => setError('')}>×</button></div>}
          {message && <div className="success-box global">{message}<button onClick={() => setMessage('')}>×</button></div>}
          <div className="stats-grid"><Stat label="Jobs" value={stats.jobs}/><Stat label="Open jobs" value={stats.open}/><Stat label="Candidates in job" value={stats.candidates}/><Stat label="Shortlisted" value={stats.shortlisted}/></div>
          {selectedJob ? <JobWorkspace job={selectedJob} candidates={candidateRows} busy={busy} onUpload={uploadResumes} onStatus={updateStatus} onSelect={setSelectedCandidate} onDelete={deleteSelectedJob}/> : <EmptyState onCreate={() => setShowCreate(true)}/>} 
        </> : <InsightsWorkspace jobs={jobs} selectedJob={selectedJob} candidates={candidateRows} onJobChange={(id)=>setSelectedJobId(id)} /> }
      </main>
    </div>
    {showCreate && <CreateJobModal onClose={() => setShowCreate(false)} onCreate={createJob} busy={creating}/>} 
    {selectedCandidate && <CandidateDrawer row={selectedCandidate} onClose={() => setSelectedCandidate(null)} />}
  </div>
}

function mergeCandidates(prev, next) {
  const map = new Map(prev.map((x) => [x.candidate_id, x]))
  next.forEach(x => map.set(x.candidate_id, x))
  return [...map.values()]
}

function Stat({ label, value }) { return <div className="stat-card"><div className="stat-label">{label}</div><div className="stat-value">{value}</div></div> }

function EmptyState({ onCreate }) { return <div className="empty-state"><div className="empty-icon">💼</div><h2>Start with a job</h2><p>Create your first job, paste the JD, then upload resumes to populate a candidate pipeline.</p><button className="primary" onClick={onCreate}>Create a job →</button></div> }

function JobWorkspace({ job, candidates, busy, onUpload, onStatus, onSelect, onDelete }) {
  const [tab, setTab] = useState('candidates')
  return <div className="job-card">
    <div className="tabs bordered"><button className={tab==='candidates'?'tab active':'tab'} onClick={()=>setTab('candidates')}>Candidates <span className="count-pill">{candidates.length}</span></button><button className={tab==='jd'?'tab active':'tab'} onClick={()=>setTab('jd')}>Job description</button><button className={tab==='settings'?'tab active':'tab'} onClick={()=>setTab('settings')}>Settings</button></div>
    {tab==='candidates' && <div className="tab-content"><div className="toolbar"><div><h3>Candidate pipeline</h3><p className="muted">Upload one or more resumes. The platform runs the existing ATS, requirements, semantic analysis and explainability pipeline.</p></div><label className={busy ? 'upload-btn disabled':'upload-btn'}>{busy ? 'Analyzing…' : '＋ Upload resumes'}<input type="file" multiple accept=".pdf,.docx,.txt,.md" disabled={busy} onChange={(e)=>{const files=[...e.target.files]; if(files.length) onUpload(files); e.target.value=''}}/></label></div><CandidateTable rows={candidates} onStatus={onStatus} onSelect={onSelect}/></div>}
    {tab==='jd' && <div className="tab-content"><div className="jd-box">{job.description}</div></div>}
    {tab==='settings' && <div className="tab-content"><div className="settings-row"><div><h3>Job status</h3><p className="muted">Closed jobs cannot accept new resume analysis.</p></div><span className={`badge ${job.status}`}>{job.status}</span></div><div className="settings-row danger-row"><div><h3>Delete this job</h3><p className="muted">This removes the job and its applications from the current account.</p></div><button className="danger" onClick={onDelete}>Delete job</button></div></div>}
  </div>
}

function CandidateTable({ rows, onStatus, onSelect }) {
  if (!rows.length) return <div className="empty-table"><div className="empty-icon small">📄</div><h3>No candidates yet</h3><p className="muted">Upload resumes to run the screening pipeline.</p></div>
  return <div className="table-wrap"><table><thead><tr><th>Candidate</th><th>ATS score</th><th>Requirement match</th><th>Status</th><th></th></tr></thead><tbody>{rows.map(row => {const a=row.analysis||{}; const ats=atsOf(a); const req=arrOf(a,'required_matched').length; const reqTotal=(a.requirement_counts?.required_total ?? a.requirements?.required_total ?? null); return <tr key={row.candidate_id}><td><button className="candidate-name" onClick={()=>onSelect(row)}>{row.name || 'Unnamed candidate'}</button><div className="candidate-meta">{row.email || row.resume_filename}</div></td><td><span className={`score ${scoreClass(ats)}`}>{ats == null ? '—' : `${ats.toFixed(1)}%`}</span></td><td>{reqTotal == null ? <span className="muted">See details</span> : <span>{req}/{reqTotal}</span>}</td><td><select value={row.status} onChange={(e)=>onStatus(row.candidate_id,e.target.value)}>{STATUS_OPTIONS.map(s=><option key={s} value={s}>{s}</option>)}</select></td><td><button className="icon-btn" onClick={()=>onSelect(row)}>View →</button></td></tr>})}</tbody></table></div>
}

function CandidateDrawer({ row, onClose }) {
  const a=row.analysis||{}
  const ats=atsOf(a)
  const scores=[
    ['Skill match', scoreOf(a,'skill_score')],
    ['Semantic match', scoreOf(a,'semantic_score')],
    ['Lexical match', scoreOf(a,'lexical_score')],
    ['Keyword match', scoreOf(a,'keyword_score')]
  ]
  const matched=arrOf(a,'matched_skills')
  const missing=arrOf(a,'missing_skills')
  const reqMatched=arrOf(a,'required_matched')
  const reqMissing=arrOf(a,'required_missing')
  const evidence=arrOf(a,'evidence', arrOf(a,'supporting_evidence', arrOf(a,'resume_evidence')))
  return <div className="drawer-backdrop" onMouseDown={(e)=>e.target===e.currentTarget&&onClose()}><aside className="drawer">
    <div className="drawer-head"><div><div className="eyebrow">CANDIDATE 360°</div><h2>{row.name || 'Unnamed candidate'}</h2><p>{row.email || row.resume_filename}</p></div><button className="close" onClick={onClose}>×</button></div>
    <div className="drawer-body">
      <div className="hero-score"><div><div className="stat-label">EXISTING ATS SCORE</div><div className={`hero-number ${scoreClass(ats)}`}>{ats == null ? '—' : `${ats.toFixed(1)}%`}</div></div><div className={`badge ${row.status}`}>{row.status}</div></div>
      <Section title="Score breakdown"><div className="score-grid">{scores.map(([label,value])=><div className="mini-score" key={label}><span>{label}</span><b>{value==null?'—':`${Number(value).toFixed(1)}%`}</b><div className="meter"><i style={{width:`${Math.max(0,Math.min(100,Number(value)||0))}%`}} /></div></div>)}</div></Section>
      <Section title="Skills"><div className="tag-group">{matched.length?matched.map(x=><span className="tag success" key={x}>✓ {x}</span>):<span className="muted">No matched skills returned.</span>}</div>{missing.length>0&&<><div className="sub-label">Not detected</div><div className="tag-group">{missing.map(x=><span className="tag danger-tag" key={x}>× {x}</span>)}</div></>}</Section>
      <Section title="Requirement coverage"><div className="coverage-list"><Coverage label="Required matched" values={reqMatched}/><Coverage label="Required missing" values={reqMissing}/></div></Section>
      <Section title="Evidence"><div className="evidence-list">{evidence.length?evidence.slice(0,8).map((item,i)=><div className="evidence" key={i}>“{textOf(item)}”</div>):<div className="muted">No evidence snippets were returned by the API for this candidate.</div>}</div></Section>
      <Section title="Contact & resume"><Info label="Phone" value={row.phone}/><Info label="LinkedIn" value={row.linkedin}/><Info label="GitHub" value={row.github}/><Info label="File" value={row.resume_filename}/></Section>
    </div>
  </aside></div>
}

function Section({title,children}){return <section className="drawer-section"><div className="section-title">{title}</div>{children}</section>}
function Coverage({label,values}){return <div className="coverage-block"><div className="sub-label">{label}</div>{values.length?<div className="tag-group">{values.map(x=><span className="tag" key={x}>{x}</span>)}</div>:<span className="muted">None detected.</span>}</div>}
function Info({label,value}){return <div className="info-row"><span>{label}</span><b>{value||'Not available'}</b></div>}

function InsightsWorkspace({ jobs, selectedJob, candidates, onJobChange }) {
  const [tab, setTab] = useState('evaluation')
  const [auditRows, setAuditRows] = useState([])
  const [runs, setRuns] = useState([])
  const [labels, setLabels] = useState({})
  const [threshold, setThreshold] = useState(50)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    api.auditLogs(100).then(setAuditRows).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!selectedJob) { setRuns([]); setLabels({}); return }
    api.evaluationRuns(selectedJob.id, 20).then(setRuns).catch((e) => setError(e.message))
    const initial = {}
    candidates.forEach((c) => { initial[c.candidate_id] = labels[c.candidate_id] || '' })
    setLabels(initial)
  }, [selectedJob?.id, candidates.length])

  async function runEvaluation() {
    if (!selectedJob) return
    const labeled = candidates.filter((c) => labels[c.candidate_id])
    if (!labeled.length) { setError('Label at least one candidate.'); return }
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await api.evaluateJob(selectedJob.id, {
        threshold: Number(threshold),
        labels: labeled.map((c) => ({ candidate_id: c.candidate_id, label: labels[c.candidate_id] }))
      })
      setRuns((prev) => [result, ...prev.filter((r) => r.id !== result.id)])
      setMessage('Evaluation run saved.')
      api.auditLogs(100).then(setAuditRows).catch(() => {})
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const latest = runs[0]
  return <>
    <div className="page-head"><div><div className="eyebrow">V14 · EVALUATION + AUDIT</div><h1>Quality & traceability</h1><p>Measure threshold behavior on recruiter-labeled examples and inspect recorded platform activity.</p></div><div className="head-actions"><select className="job-picker" value={selectedJob?.id || ''} onChange={(e)=>onJobChange(e.target.value)}>{jobs.length ? jobs.map(j=><option key={j.id} value={j.id}>{j.title}</option>) : <option value="">No jobs</option>}</select></div></div>
    {error && <div className="error-box global">{error}<button onClick={() => setError('')}>×</button></div>}
    {message && <div className="success-box global">{message}<button onClick={() => setMessage('')}>×</button></div>}
    <div className="v14-notice">🛡️ <b>Decision-support telemetry:</b> scores and detected requirements are signals for recruiter review. A missing signal does not prove a candidate lacks a skill.</div>
    <div className="tabs bordered v14-tabs"><button className={tab==='evaluation'?'tab active':'tab'} onClick={()=>setTab('evaluation')}>Evaluation</button><button className={tab==='audit'?'tab active':'tab'} onClick={()=>setTab('audit')}>Audit log <span className="count-pill">{auditRows.length}</span></button></div>
    {tab==='evaluation' ? <EvaluationPanel candidates={candidates} labels={labels} setLabels={setLabels} threshold={threshold} setThreshold={setThreshold} latest={latest} runs={runs} busy={busy} onRun={runEvaluation}/> : <AuditPanel rows={auditRows}/>} 
  </>
}

function EvaluationPanel({ candidates, labels, setLabels, threshold, setThreshold, latest, runs, busy, onRun }) {
  return <div className="v14-panel">
    {!candidates.length ? <div className="empty-table"><div className="empty-icon small">📊</div><h3>No candidates in this job</h3><p className="muted">Upload resumes first, then label a set of candidates to evaluate threshold behavior.</p></div> : <>
      <div className="evaluation-controls"><div><div className="section-title">Evaluation threshold</div><div className="muted small-text">Compare the existing ATS score with a configurable threshold. This is a diagnostic, not an automatic hiring rule.</div></div><div className="threshold-control"><input type="number" min="0" max="100" step="1" value={threshold} onChange={(e)=>setThreshold(e.target.value)}/><span>%</span><button className="primary" disabled={busy} onClick={onRun}>{busy ? 'Running…' : 'Run evaluation'}</button></div></div>
      <div className="table-wrap"><table><thead><tr><th>Candidate</th><th>ATS score</th><th>Human label</th></tr></thead><tbody>{candidates.map(c=>{const ats=atsOf(c.analysis||{}); return <tr key={c.candidate_id}><td><b>{c.name || 'Unnamed candidate'}</b><div className="candidate-meta">{c.resume_filename}</div></td><td><span className={`score ${scoreClass(ats)}`}>{ats==null?'—':`${ats.toFixed(1)}%`}</span></td><td><select value={labels[c.candidate_id] || ''} onChange={(e)=>setLabels(prev=>({...prev,[c.candidate_id]:e.target.value}))}><option value="">Exclude</option><option value="qualified">Qualified</option><option value="not_qualified">Not qualified</option></select></td></tr>})}</tbody></table></div>
    </>}
    {latest && <div className="metric-grid-v14"><Metric label="Accuracy" value={latest.metrics.accuracy}/><Metric label="Precision" value={latest.metrics.precision}/><Metric label="Recall" value={latest.metrics.recall}/><Metric label="F1" value={latest.metrics.f1}/><Metric label="Sample size" value={latest.metrics.sample_size} suffix=""/><Metric label="Threshold" value={latest.metrics.threshold} suffix="%"/></div>}
    {latest && <div className="confusion"><div><span>True positive</span><b>{latest.metrics.true_positive}</b></div><div><span>True negative</span><b>{latest.metrics.true_negative}</b></div><div><span>False positive</span><b>{latest.metrics.false_positive}</b></div><div><span>False negative</span><b>{latest.metrics.false_negative}</b></div></div>}
    {runs.length>1 && <Section title="Previous evaluation runs"><div className="run-list">{runs.map(r=><div className="run-row" key={r.id}><span>{new Date(r.created_at).toLocaleString()}</span><b>{r.metrics.sample_size} labeled · threshold {r.metrics.threshold}% · F1 {r.metrics.f1 == null ? '—' : `${Number(r.metrics.f1).toFixed(1)}%`}</b></div>)}</div></Section>}
  </div>
}

function Metric({label,value,suffix='%'}){ return <div className="stat-card"><div className="stat-label">{label}</div><div className="stat-value">{value==null?'—':`${Number(value).toFixed(1)}${suffix}`}</div></div> }

function AuditPanel({rows}) { return <div className="v14-panel"><div className="toolbar"><div><h3>Audit history</h3><p className="muted">Successful recruiter actions recorded by the V14 backend. Resume contents and passwords are not stored in audit metadata.</p></div></div>{rows.length?<div className="audit-list">{rows.map(r=><div className="audit-row" key={r.id}><div><b>{r.action.replaceAll('_',' ')}</b><div className="candidate-meta">{r.entity_type}{r.entity_id ? ` · ${r.entity_id.slice(0, 8)}` : ''}</div></div><div className="audit-meta"><span>{new Date(r.created_at).toLocaleString()}</span><code>{JSON.stringify(r.metadata || {})}</code></div></div>)}</div>:<div className="empty-table"><h3>No audit events yet</h3><p className="muted">New V14 activity will appear here.</p></div>}</div>}

function CreateJobModal({onClose,onCreate,busy}){
  const [form,setForm]=useState({title:'',company:'',description:''})
  return <div className="modal-backdrop"><div className="modal"><div className="modal-head"><div><div className="eyebrow">NEW JOB</div><h2>Create a recruiting job</h2></div><button className="close" onClick={onClose}>×</button></div><form onSubmit={(e)=>{e.preventDefault();onCreate(form)}} className="stack gap-md"><Field label="Job title"><input value={form.title} onChange={e=>setForm({...form,title:e.target.value})} required placeholder="AI Engineer"/></Field><Field label="Company"><input value={form.company} onChange={e=>setForm({...form,company:e.target.value})} placeholder="Your company"/></Field><Field label="Job description"><textarea value={form.description} onChange={e=>setForm({...form,description:e.target.value})} required minLength={20} rows={10} placeholder="Paste the full job description here…"/></Field><div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button type="submit" className="primary" disabled={busy}>{busy?'Creating…':'Create job'}</button></div></form></div></div>
}

export default App
