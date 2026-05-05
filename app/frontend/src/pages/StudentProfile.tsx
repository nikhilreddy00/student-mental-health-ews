import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell, CartesianGrid,
} from 'recharts'
import { getStudent, getHighRiskIds, postNarrative } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import PageHeader from '../components/layout/PageHeader'
import RiskBadge from '../components/ui/RiskBadge'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'

function RiskGauge({ score }: { score: number }) {
  const angle = 180 - score * 180 // 180° = 0, 0° = 1.0
  const cx = 100; const cy = 90; const r = 75
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const arcPath = (startDeg: number, endDeg: number, color: string) => {
    const s = toRad(startDeg); const e = toRad(endDeg)
    const x1 = cx + r * Math.cos(s); const y1 = cy - r * Math.sin(s)
    const x2 = cx + r * Math.cos(e); const y2 = cy - r * Math.sin(e)
    return <path d={`M${x1},${y1} A${r},${r} 0 0,0 ${x2},${y2}`} fill="none" stroke={color} strokeWidth={16} strokeLinecap="butt" />
  }
  const needleRad = toRad(angle)
  const nx = cx + (r - 8) * Math.cos(needleRad)
  const ny = cy - (r - 8) * Math.sin(needleRad)

  return (
    <svg viewBox="0 0 200 105" className="w-full max-w-[220px] mx-auto">
      {arcPath(0, 60, '#10b981')}
      {arcPath(60, 120, '#f59e0b')}
      {arcPath(120, 180, '#ef4444')}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="white" strokeWidth={2.5} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={5} fill="white" />
      <text x={cx} y={cy + 16} textAnchor="middle" fill="white" fontSize={14} fontWeight="bold">{score.toFixed(3)}</text>
      <text x={20} y={100} fill="#10b981" fontSize={10}>Low</text>
      <text x={85} y={18} fill="#f59e0b" fontSize={10}>Med</text>
      <text x={155} y={100} fill="#ef4444" fontSize={10}>High</text>
    </svg>
  )
}

export default function StudentProfile() {
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [inputVal, setInputVal] = useState('')

  const { data: ids } = useQuery({
    queryKey: ['high-risk-ids', true],
    queryFn: () => getHighRiskIds(true),
    staleTime: 30 * 60 * 1000,
  })

  const { data: student, isLoading, error } = useQuery({
    queryKey: ['student', selectedId],
    queryFn: () => getStudent(selectedId!),
    enabled: selectedId !== null,
    staleTime: 10 * 60 * 1000,
  })

  const narrativeMutation = useMutation({
    mutationFn: () => postNarrative(selectedId!),
  })

  function handleSearch() {
    const n = parseInt(inputVal)
    if (!isNaN(n)) { setSelectedId(n); setInputVal('') }
  }

  return (
    <div>
      <PageHeader title="Student Profile" subtitle="Individual risk profile, SHAP explainability, and counselor actions" />
      <div className="p-6 space-y-6">

        {/* Student selector */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <div className="flex flex-wrap gap-3 items-end">
              <div className="form-control">
                <label className="label py-0 mb-1"><span className="label-text text-xs">Select High-Risk Student</span></label>
                <select
                  className="select select-sm select-bordered w-56"
                  value={selectedId ?? ''}
                  onChange={e => setSelectedId(Number(e.target.value))}
                >
                  <option value="">-- Choose student ID --</option>
                  {ids?.ids.slice(0, 200).map(id => <option key={id} value={id}>{id}</option>)}
                </select>
              </div>
              <div className="form-control">
                <label className="label py-0 mb-1"><span className="label-text text-xs">Or enter ID directly</span></label>
                <div className="join">
                  <input
                    type="number"
                    className="input input-sm input-bordered join-item w-36"
                    placeholder="Student ID…"
                    value={inputVal}
                    onChange={e => setInputVal(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSearch()}
                  />
                  <button className="btn btn-sm btn-primary join-item" onClick={handleSearch}>Look up</button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {!selectedId && (
          <div className="text-center py-16 opacity-40">Select a student to view their profile</div>
        )}

        {isLoading && <LoadingSpinner label="Loading student profile…" />}
        {error && <ErrorAlert message="Student not found or failed to load." />}

        {student && (
          <>
            {/* Header stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="stat bg-base-200 rounded-box shadow-sm">
                <div className="stat-title text-xs uppercase opacity-70">Risk Score</div>
                <div className="stat-value text-2xl font-bold text-error">{student.risk_score.toFixed(3)}</div>
              </div>
              <div className="stat bg-base-200 rounded-box shadow-sm">
                <div className="stat-title text-xs uppercase opacity-70">Risk Tier</div>
                <div className="stat-value text-2xl"><RiskBadge tier={student.risk_tier} size="md" /></div>
              </div>
              <div className="stat bg-base-200 rounded-box shadow-sm">
                <div className="stat-title text-xs uppercase opacity-70">Module</div>
                <div className="stat-value text-2xl font-bold">{student.code_module}</div>
              </div>
              <div className="stat bg-base-200 rounded-box shadow-sm">
                <div className="stat-title text-xs uppercase opacity-70">Student ID</div>
                <div className="stat-value text-2xl font-mono">{student.student_id}</div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Gauge */}
              <div className="card bg-base-200 shadow-sm">
                <div className="card-body items-center p-4">
                  <h3 className="card-title text-sm">Risk Gauge</h3>
                  <RiskGauge score={student.risk_score} />
                  <div className="text-xs opacity-60 mt-1">0 = No Risk · 1 = Maximum Risk</div>
                </div>
              </div>

              {/* Demographics */}
              <div className="card bg-base-200 shadow-sm">
                <div className="card-body p-4">
                  <h3 className="card-title text-sm">Demographics</h3>
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-sm">
                    {[
                      ['Gender', student.gender],
                      ['Age Band', student.age_band],
                      ['Region', student.region],
                      ['IMD Band', student.imd_band],
                      ['Education', student.highest_education],
                      ['Disability', student.disability],
                      ['Prev Attempts', student.num_of_prev_attempts],
                      ['Credits', student.studied_credits],
                    ].map(([k, v]) => (
                      <>
                        <dt key={`k-${k}`} className="opacity-60 text-xs">{k}</dt>
                        <dd key={`v-${k}`} className="font-medium text-xs">{String(v) || '—'}</dd>
                      </>
                    ))}
                  </dl>
                </div>
              </div>

              {/* Behavioral */}
              <div className="card bg-base-200 shadow-sm">
                <div className="card-body p-4">
                  <h3 className="card-title text-sm">Behavioral Metrics</h3>
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-sm">
                    {[
                      ['Engagement Span', `${student.engagement_span.toFixed(0)} days`],
                      ['Active Days', student.active_days.toFixed(0)],
                      ['Mean Score', `${student.mean_score.toFixed(1)}/100`],
                      ['Submission Rate', `${(student.submission_rate * 100).toFixed(1)}%`],
                      ['Engagement Decline', student.engagement_decline.toFixed(3)],
                      ['Dropout Modules', student.dropout_modules],
                    ].map(([k, v]) => (
                      <>
                        <dt key={`k-${k}`} className="opacity-60 text-xs">{k}</dt>
                        <dd key={`v-${k}`} className="font-medium text-xs">{String(v)}</dd>
                      </>
                    ))}
                  </dl>
                  <div className="mt-3 space-y-1">
                    {[student.top_reason, student.reason_2, student.reason_3].filter(Boolean).map((r, i) => (
                      <div key={i} className="badge badge-error badge-sm block text-xs">{i + 1}. {r}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* SHAP waterfall */}
            {student.shap_waterfall.length > 0 && (
              <div className="card bg-base-200 shadow-sm">
                <div className="card-body p-4">
                  <h3 className="card-title text-sm">SHAP Feature Contributions</h3>
                  <p className="text-xs opacity-60 mb-2">Red = increases risk · Blue = decreases risk · Length = magnitude</p>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart
                      layout="vertical"
                      data={[...student.shap_waterfall].sort((a, b) => b.shap_value - a.shap_value)}
                      margin={{ top: 4, right: 16, left: 140, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                      <XAxis type="number" tick={{ fontSize: 10 }} />
                      <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={140} />
                      <ReferenceLine x={0} stroke="#666" />
                      <Tooltip formatter={(v: number) => [v.toFixed(4), 'SHAP Value']} contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                      <Bar dataKey="shap_value" radius={[0, 3, 3, 0]}>
                        {student.shap_waterfall.map((entry, i) => (
                          <Cell key={i} fill={entry.direction === 'positive' ? '#ef4444' : '#3b82f6'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* AI Narrative */}
            <div className="card bg-base-200 shadow-sm">
              <div className="card-body p-4">
                <div className="flex items-center justify-between">
                  <h3 className="card-title text-sm">AI Risk Narrative</h3>
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() => narrativeMutation.mutate()}
                    disabled={narrativeMutation.isPending}
                  >
                    {narrativeMutation.isPending ? <span className="loading loading-spinner loading-xs" /> : 'Generate'}
                  </button>
                </div>
                {narrativeMutation.data && (
                  <div className="alert alert-info text-sm mt-3">{narrativeMutation.data.narrative}</div>
                )}
              </div>
            </div>

            {/* Counselor action */}
            <div className={`alert ${student.risk_tier === 'High' ? 'alert-error' : student.risk_tier === 'Medium' ? 'alert-warning' : 'alert-success'}`}>
              <span className="font-semibold text-sm">Recommended Action:</span>
              <span className="text-sm">{student.suggested_action.message}</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
