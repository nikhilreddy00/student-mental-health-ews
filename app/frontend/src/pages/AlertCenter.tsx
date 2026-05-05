import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { AlertCircle, Download } from 'lucide-react'
import { getAlerts } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import { useFilterStore } from '../store/filterStore'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import RiskBadge from '../components/ui/RiskBadge'
import RiskProgress from '../components/ui/RiskProgress'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'
import type { AlertStudent } from '../types'

export default function AlertCenter() {
  const { module } = useFilterStore()
  const [page, setPage] = useState(1)
  const PAGE_SIZE = 50

  const { data, isLoading, error } = useQuery({
    queryKey: ['alerts', module, page],
    queryFn: () => getAlerts({ module, tiers: ['High'], page, page_size: PAGE_SIZE }),
    staleTime: 5 * 60 * 1000,
  })

  function downloadCSV() {
    if (!data) return
    const headers = 'student_id,risk_score,risk_tier,module,top_reason,reason_2,reason_3'
    const rows = data.students.map((s: AlertStudent) =>
      `${s.id_student},${s.risk_score},${s.risk_tier},${s.code_module},"${s.top_reason}","${s.reason_2}","${s.reason_3}"`
    )
    const csv = [headers, ...rows].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    const a = document.createElement('a'); a.href = url; a.download = 'high_risk_students.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  if (isLoading) return <LoadingSpinner label="Loading alert center…" />
  if (error || !data) return <ErrorAlert message="Failed to load alerts." />

  const totalPages = Math.ceil(data.total / PAGE_SIZE)

  return (
    <div>
      <PageHeader title="Alert Center" subtitle="High-risk students requiring counselor attention" />
      <div className="p-6 space-y-6">

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 panel px-4 py-2 border-error/30">
            <AlertCircle size={14} className="text-error shrink-0" />
            <span className="text-sm font-semibold text-error">{data.total.toLocaleString()} students require attention</span>
          </div>
          <button className="btn btn-sm btn-outline gap-1.5" onClick={downloadCSV}>
            <Download size={13} /> Download CSV
          </button>
        </div>

        <StatCard
          label="High Risk Students"
          value={data.total.toLocaleString()}
          accent="high"
          desc="Sorted by risk score descending"
          mono
        />

        {/* Student table */}
        <div className="card bg-base-200 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="table table-sm text-sm">
              <thead>
                <tr className="bg-base-300 text-xs uppercase tracking-wide">
                  <th>Student ID</th>
                  <th>Risk Score</th>
                  <th>Tier</th>
                  <th>Module</th>
                  <th>Primary Reason</th>
                  <th>Reason 2</th>
                  <th>Reason 3</th>
                </tr>
              </thead>
              <tbody>
                {data.students.map((s: AlertStudent) => (
                  <tr key={s.id_student} className="hover">
                    <td className="font-mono font-semibold">{s.id_student}</td>
                    <td><RiskProgress value={s.risk_score} /></td>
                    <td><RiskBadge tier={s.risk_tier} size="sm" /></td>
                    <td><span className="badge badge-ghost badge-sm">{s.code_module}</span></td>
                    <td className="max-w-32 truncate text-xs">{s.top_reason}</td>
                    <td className="max-w-28 truncate text-xs opacity-70">{s.reason_2}</td>
                    <td className="max-w-28 truncate text-xs opacity-70">{s.reason_3}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Pagination */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-base-300">
            <span className="text-xs opacity-60">Page {page} of {totalPages} · {data.total.toLocaleString()} total</span>
            <div className="join">
              <button className="join-item btn btn-xs" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>«</button>
              <button className="join-item btn btn-xs btn-active">{page}</button>
              <button className="join-item btn btn-xs" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>»</button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Risk factors */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">Top Risk Factors (High-Risk Cohort)</h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart layout="vertical" data={data.risk_factors} margin={{ top: 4, right: 8, left: 120, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="factor" tick={{ fontSize: 10 }} width={120} />
                  <Tooltip contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                  <Bar dataKey="count" fill="#ef4444" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Behavioral comparison */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">High-Risk vs Low-Risk Behavioral Profile</h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart
                  data={data.behavioral_comparison.features.map((f, i) => ({
                    feature: f.replace(/_/g, ' '),
                    'High Risk': data.behavioral_comparison.high_risk_avgs[i],
                    'Low Risk': data.behavioral_comparison.low_risk_avgs[i],
                  }))}
                  margin={{ top: 4, right: 8, left: 0, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="feature" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                  <Bar dataKey="High Risk" fill="#ef4444" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Low Risk" fill="#10b981" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
