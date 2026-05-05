import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Legend,
} from 'recharts'
import { CheckCircle, Download } from 'lucide-react'
import { getTemporalAnalysis } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import { useSSE } from '../hooks/useSSE'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import StreamingText from '../components/ui/StreamingText'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'

const LINE_COLORS = ['#6366f1', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#a855f7', '#ef4444', '#84cc16', '#f97316', '#14b8a6']

export default function EarlyWarningAnalysis() {
  const [earlyThreshold, setEarlyThreshold] = useState(0.50)
  const [consecutiveWeeks, setConsecutiveWeeks] = useState(2)
  const [narrative, setNarrative] = useState('')
  const { isStreaming, startStream } = useSSE()

  const { data, isLoading, error } = useQuery({
    queryKey: ['temporal', earlyThreshold, consecutiveWeeks],
    queryFn: () => getTemporalAnalysis(earlyThreshold, consecutiveWeeks),
    staleTime: 5 * 60 * 1000,
  })

  async function generateNarrative() {
    setNarrative('')
    await startStream(
      '/api/temporal-analysis/narrative',
      { early_threshold: earlyThreshold, consecutive_weeks: consecutiveWeeks },
      (event) => {
        if (event.chunk) setNarrative(prev => prev + event.chunk)
      },
    )
  }

  if (isLoading) return <LoadingSpinner label="Computing intervention delay analysis…" />
  if (error || !data) return <ErrorAlert message="Failed to load temporal analysis." />

  const { metrics, students, trajectories } = data

  // Chart data: show all trajectories
  const allWeeks = trajectories.reduce((max, t) => Math.max(max, ...t.weeks), 0)
  const chartData = Array.from({ length: allWeeks + 1 }, (_, week) => {
    const point: Record<string, number | null | string> = { week }
    trajectories.forEach(t => {
      const wi = t.weeks.indexOf(week)
      point[`s${t.student_id}`] = wi >= 0 ? t.risk_scores[wi] : null
    })
    return point
  })

  function downloadCSV() {
    const headers = 'student_id,module,top_reason,final_risk,early_signal_week,actual_alert_week,delay_weeks'
    const rows = students.map(s =>
      `${s.student_id},${s.module},"${s.top_reason}",${s.final_risk},${s.early_signal_week ?? ''},${s.actual_alert_week ?? ''},${s.delay_weeks ?? ''}`
    )
    const csv = [headers, ...rows].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    const a = document.createElement('a'); a.href = url; a.download = 'temporal_analysis.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <PageHeader title="Early Warning Analysis" subtitle="Quantifying intervention delay: how much earlier can we act?" />
      <div className="p-6 space-y-6">

        {/* Sliders */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm mb-3">Analysis Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium">Early Signal Threshold: {earlyThreshold.toFixed(2)}</label>
                <input
                  type="range" className="range range-primary range-sm mt-1 w-full"
                  min={0.40} max={0.65} step={0.01}
                  value={earlyThreshold}
                  onChange={e => setEarlyThreshold(Number(e.target.value))}
                />
                <div className="flex justify-between text-xs opacity-50 mt-0.5"><span>0.40</span><span>0.65</span></div>
              </div>
              <div>
                <label className="text-sm font-medium">Consecutive Weeks Required: {consecutiveWeeks}</label>
                <input
                  type="range" className="range range-secondary range-sm mt-1 w-full"
                  min={1} max={4} step={1}
                  value={consecutiveWeeks}
                  onChange={e => setConsecutiveWeeks(Number(e.target.value))}
                />
                <div className="flex justify-between text-xs opacity-50 mt-0.5"><span>1</span><span>4</span></div>
              </div>
            </div>
          </div>
        </div>

        {/* Metric cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Mean Delay" value={`${metrics.mean_delay} wks`} accent="blue" desc="Average intervention window" mono />
          <StatCard label="Median Delay" value={`${metrics.median_delay} wks`} mono />
          <StatCard label="Students w/ Window" value={metrics.n_with_delay} desc={`of ${metrics.n_total} total`} mono />
          <StatCard label="Max Delay Observed" value={`${metrics.max_delay} wks`} accent="med" mono />
        </div>

        {metrics.mean_delay > 0 && (
          <div className="alert alert-success text-sm flex items-center gap-2">
            <CheckCircle size={15} className="shrink-0" />
            System could identify {metrics.n_with_delay} at-risk students an average of <strong>{metrics.mean_delay} weeks earlier</strong> using the early signal threshold of {earlyThreshold}.
          </div>
        )}

        {/* Trajectory chart */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm">Risk Trajectories with Early Signal Markers</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="week" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v: number) => [v?.toFixed(4) ?? '—', 'Risk Score']} contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.cursor} />
                <ReferenceLine y={earlyThreshold} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: `Early Signal (${earlyThreshold})`, position: 'insideTopLeft', fontSize: 9, fill: '#f59e0b' }} />
                <ReferenceLine y={0.66} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Alert (0.66)', position: 'insideTopRight', fontSize: 9, fill: '#ef4444' }} />
                <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                {trajectories.map((t, i) => (
                  <Line
                    key={t.student_id}
                    type="monotone"
                    dataKey={`s${t.student_id}`}
                    name={`${t.student_id}`}
                    stroke={LINE_COLORS[i % LINE_COLORS.length]}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Per-student delay table */}
        <div className="card bg-base-200 shadow-sm overflow-hidden">
          <div className="card-body p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="card-title text-sm">Per-Student Delay Analysis</h3>
              <button className="btn btn-xs btn-outline gap-1" onClick={downloadCSV}><Download size={11} /> CSV</button>
            </div>
            <div className="overflow-x-auto max-h-64">
              <table className="table table-sm text-xs">
                <thead className="sticky top-0 bg-base-300">
                  <tr><th>Student ID</th><th>Module</th><th>Top Reason</th><th>Final Risk</th><th>Early Signal Week</th><th>Alert Week</th><th>Delay</th></tr>
                </thead>
                <tbody>
                  {students.map(s => (
                    <tr key={s.student_id} className="hover">
                      <td className="font-mono">{s.student_id}</td>
                      <td>{s.module}</td>
                      <td className="max-w-28 truncate">{s.top_reason}</td>
                      <td>{s.final_risk.toFixed(3)}</td>
                      <td>{s.early_signal_week ?? '—'}</td>
                      <td>{s.actual_alert_week ?? '—'}</td>
                      <td className={s.delay_weeks != null ? 'font-bold text-primary' : 'opacity-40'}>
                        {s.delay_weeks != null ? `${s.delay_weeks}w` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Research narrative */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="card-title text-sm">Research Narrative</h3>
              <button
                className="btn btn-sm btn-primary"
                onClick={generateNarrative}
                disabled={isStreaming}
              >
                {isStreaming ? <><span className="loading loading-spinner loading-xs" /> Generating…</> : 'Generate Narrative'}
              </button>
            </div>
            {narrative ? (
              <StreamingText text={narrative} isStreaming={isStreaming} className="text-sm" />
            ) : (
              <p className="text-xs opacity-40 italic">Click "Generate Narrative" to produce an academic-style research finding.</p>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}
