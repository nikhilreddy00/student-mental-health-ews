import { useState, useMemo, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Legend,
} from 'recharts'
import { AlertTriangle } from 'lucide-react'
import { getRiskMonitor } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import type { RiskMonitorData } from '../types'
import { useSSE } from '../hooks/useSSE'
import PageHeader from '../components/layout/PageHeader'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'
import type { Trajectory } from '../types'

const LINE_COLORS = ['#6366f1', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#a855f7', '#ef4444', '#84cc16', '#f97316', '#14b8a6']

export default function LiveRiskMonitor() {
  const [currentWeek, setCurrentWeek] = useState(0)
  const [selectedStudents, setSelectedStudents] = useState<number[]>([])
  const [alertTexts, setAlertTexts] = useState<Record<string, string>>({})
  const { startStream } = useSSE()

  const { data, isLoading, error } = useQuery<RiskMonitorData>({
    queryKey: ['risk-monitor'],
    queryFn: getRiskMonitor,
    staleTime: 30 * 60 * 1000,
  })

  useEffect(() => {
    if (data?.students.length) {
      setSelectedStudents(data.students.slice(0, 3).map(s => s.student_id))
      setCurrentWeek(data.max_week)
    }
  }, [data])

  const chartData = useMemo(() => {
    if (!data) return []
    const allWeeks = Array.from({ length: currentWeek + 1 }, (_, i) => i)
    return allWeeks.map(week => {
      const point: Record<string, number | null | string> = { week }
      for (const sid of selectedStudents) {
        const traj = data.students.find(s => s.student_id === sid)
        const idx = traj?.weeks.indexOf(week) ?? -1
        point[`s${sid}`] = idx >= 0 ? traj!.risk_scores[idx] : null
      }
      return point
    })
  }, [data, selectedStudents, currentWeek])

  function toggleStudent(id: number) {
    setSelectedStudents(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  async function generateAlert(traj: Trajectory, week: number, prev: number, curr: number) {
    const key = `${traj.student_id}-${week}`
    setAlertTexts(prev => ({ ...prev, [key]: '' }))
    await startStream(
      '/api/risk-alert',
      { student_id: traj.student_id, week, prev_score: prev, curr_score: curr, top_reason: traj.top_reason },
      (event) => {
        if (event.chunk) {
          setAlertTexts(prev => ({ ...prev, [key]: (prev[key] ?? '') + event.chunk }))
        }
      },
    )
  }

  if (isLoading) return <LoadingSpinner label="Loading risk trajectories… (may take 30-60s on first load)" />
  if (error || !data) return <ErrorAlert message="Failed to load risk monitor data." />

  // Detect threshold crossings at currentWeek
  const crossings: { traj: Trajectory; week: number; prev: number; curr: number }[] = []
  for (const sid of selectedStudents) {
    const traj = data.students.find(s => s.student_id === sid)
    if (!traj) continue
    const wi = traj.weeks.indexOf(currentWeek)
    if (wi > 0) {
      const curr = traj.risk_scores[wi]
      const prev = traj.risk_scores[wi - 1]
      if (prev < data.threshold && curr >= data.threshold) {
        crossings.push({ traj, week: currentWeek, prev, curr })
      }
    }
  }

  return (
    <div>
      <PageHeader title="Live Risk Monitor" subtitle="Week-by-week risk trajectories with threshold alerts" />
      <div className="p-6 space-y-6">

        {/* Week slider */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <div className="flex items-center gap-4">
              <label className="text-sm font-semibold whitespace-nowrap">Week {currentWeek}</label>
              <input
                type="range"
                className="range range-primary flex-1"
                min={0}
                max={data.max_week}
                value={currentWeek}
                onChange={e => setCurrentWeek(Number(e.target.value))}
              />
              <span className="text-xs opacity-60">Max: {data.max_week}</span>
            </div>
          </div>
        </div>

        {/* Student checkboxes */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm mb-2">Select Students to Track</h3>
            <div className="flex flex-wrap gap-2">
              {data.students.map((s, i) => (
                <label key={s.student_id} className="flex items-center gap-1.5 cursor-pointer bg-base-300 rounded px-2 py-1">
                  <input
                    type="checkbox"
                    className="checkbox checkbox-xs"
                    style={{ accentColor: LINE_COLORS[i % LINE_COLORS.length] }}
                    checked={selectedStudents.includes(s.student_id)}
                    onChange={() => toggleStudent(s.student_id)}
                  />
                  <span className="text-xs font-mono">{s.student_id}</span>
                  <span className="badge badge-ghost badge-xs">{s.module}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Line chart */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm">Risk Trajectories</h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="week" tick={{ fontSize: 10 }} label={{ value: 'Week', position: 'insideBottomRight', offset: -4, fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} tickFormatter={v => v.toFixed(1)} />
                <Tooltip formatter={(v: number) => [v?.toFixed(4) ?? '—', 'Risk Score']} contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.cursor} />
                <ReferenceLine y={data.threshold} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Alert Threshold (0.66)', position: 'insideTopRight', fontSize: 10, fill: '#ef4444' }} />
                <Legend iconSize={10} wrapperStyle={{ fontSize: 10 }} />
                {selectedStudents.map((sid, i) => (
                  <Line
                    key={sid}
                    type="monotone"
                    dataKey={`s${sid}`}
                    name={`Student ${sid}`}
                    stroke={LINE_COLORS[i % LINE_COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Crossing alerts */}
        {crossings.map(({ traj, week, prev, curr }) => {
          const key = `${traj.student_id}-${week}`
          return (
            <div key={key} className="alert alert-warning">
              <div className="flex-1">
                <span className="flex items-center gap-1.5 font-semibold"><AlertTriangle size={14} className="shrink-0" /> Threshold crossed!</span>
                Student {traj.student_id} ({traj.module}) —
                Week {week}: {prev.toFixed(3)} → {curr.toFixed(3)} · Concern: {traj.top_reason}
                {alertTexts[key] && <p className="text-sm mt-1">{alertTexts[key]}</p>}
              </div>
              <button
                className="btn btn-xs btn-warning"
                onClick={() => generateAlert(traj, week, prev, curr)}
              >
                Generate Alert
              </button>
            </div>
          )
        })}

        {/* Summary table */}
        <div className="card bg-base-200 shadow-sm overflow-hidden">
          <div className="card-body p-4">
            <h3 className="card-title text-sm">Status at Week {currentWeek}</h3>
            <div className="overflow-x-auto">
              <table className="table table-sm text-sm">
                <thead><tr><th>Student ID</th><th>Module</th><th>Risk at Week</th><th>Top Concern</th></tr></thead>
                <tbody>
                  {data.students.filter(s => selectedStudents.includes(s.student_id)).map(s => {
                    const wi = s.weeks.indexOf(currentWeek)
                    const score = wi >= 0 ? s.risk_scores[wi] : null
                    return (
                      <tr key={s.student_id}>
                        <td className="font-mono">{s.student_id}</td>
                        <td>{s.module}</td>
                        <td>{score != null ? score.toFixed(3) : '—'}</td>
                        <td className="text-xs opacity-70">{s.top_reason}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
