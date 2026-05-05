import { useQuery } from '@tanstack/react-query'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { getOverview } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import { useFilterStore } from '../store/filterStore'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'

const TIER_COLORS: Record<string, string> = {
  High: '#ef4444', Medium: '#f59e0b', Low: '#10b981',
}

function PieTooltip({ active, payload }: { active?: boolean; payload?: Array<{ name: string; value: number; payload: { percent?: number } }> }) {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0]
  const total = payload[0].payload as { percent?: number }
  const pct = total.percent !== undefined ? (total.percent * 100).toFixed(1) : ''
  return (
    <div style={{
      background: '#1a2235',
      border: '1px solid #334155',
      borderRadius: 6,
      padding: '6px 12px',
      fontSize: 12,
      color: '#e2e8f0',
      boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
      pointerEvents: 'none',
    }}>
      <span style={{ color: TIER_COLORS[name] ?? '#e2e8f0', fontWeight: 700 }}>{name}</span>
      <span style={{ color: '#94a3b8', marginLeft: 6 }}>{value.toLocaleString()} students</span>
      {pct && <span style={{ color: '#cbd5e1', marginLeft: 6 }}>({pct}%)</span>}
    </div>
  )
}

export default function Overview() {
  const { module, tiers } = useFilterStore()
  const { data, isLoading, error } = useQuery({
    queryKey: ['overview', module, tiers],
    queryFn: () => getOverview(module, tiers),
    staleTime: 5 * 60 * 1000,
  })

  if (isLoading) return <LoadingSpinner label="Loading overview…" />
  if (error || !data) return <ErrorAlert message="Failed to load overview data." />

  // Histogram data flattened for recharts
  const histData = Array.from({ length: 20 }, (_, i) => {
    const obj: Record<string, number | string> = { bin: (i * 5).toString() }
    for (const tier of ['High', 'Medium', 'Low']) {
      const b = data.risk_histogram[tier]?.[i]
      obj[tier] = b?.count ?? 0
    }
    return obj
  })

  const moduleData = data.module_breakdown.map(m => ({
    ...m,
    fill: m.high_risk_pct > 50 ? '#ef4444' : m.high_risk_pct > 30 ? '#f59e0b' : '#10b981',
  }))

  return (
    <div>
      <PageHeader title="Dashboard Overview" subtitle="Cohort-level mental health risk summary" />
      <div className="p-6 space-y-6">

        {/* KPI cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard label="Total Enrollments" value={data.total.toLocaleString()} accent="blue" mono desc="student-module records" />
          <StatCard label="High Risk" value={data.high.toLocaleString()} accent="high" desc={`${data.high_pct}%`} mono />
          <StatCard label="Medium Risk" value={data.medium.toLocaleString()} accent="med" desc={`${data.medium_pct}%`} mono />
          <StatCard label="Low Risk" value={data.low.toLocaleString()} accent="low" desc={`${data.low_pct}%`} mono />
          <StatCard label="Avg Risk Score" value={data.avg_risk_score.toFixed(3)} desc="0–1 scale" mono />
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Pie */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">Risk Tier Distribution</h3>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={data.tier_distribution}
                    dataKey="count"
                    nameKey="tier"
                    cx="50%"
                    cy="50%"
                    outerRadius={70}
                    label={false}
                    labelLine={false}
                  >
                    {data.tier_distribution.map(entry => (
                      <Cell key={entry.tier} fill={TIER_COLORS[entry.tier]} />
                    ))}
                  </Pie>
                  <Tooltip content={<PieTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              {/* Legend row — kept outside the chart so labels never overlap */}
              <div className="flex justify-center gap-4 mt-1">
                {data.tier_distribution.map(entry => {
                  const total = data.tier_distribution.reduce((s, e) => s + e.count, 0)
                  const pct = total > 0 ? ((entry.count / total) * 100).toFixed(0) : '0'
                  return (
                    <div key={entry.tier} className="flex items-center gap-1.5">
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: TIER_COLORS[entry.tier], display: 'inline-block', flexShrink: 0 }} />
                      <span style={{ fontSize: 12, color: TIER_COLORS[entry.tier], fontWeight: 700 }}>{entry.tier}</span>
                      <span style={{ fontSize: 12, color: '#94a3b8' }}>{pct}%</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Risk histogram */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">Risk Score Distribution</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={histData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={4} tickFormatter={v => `0.${v}`} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                  {['High', 'Medium', 'Low'].map(t => (
                    <Bar key={t} dataKey={t} stackId="a" fill={TIER_COLORS[t]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top reasons */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">Top Risk Reasons</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart layout="vertical" data={data.top_reasons} margin={{ top: 4, right: 8, left: 80, bottom: 0 }}>
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="reason" tick={{ fontSize: 10 }} width={80} />
                  <Tooltip contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                  <Bar dataKey="count" fill="#ef4444" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Module breakdown */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm">High-Risk % by Module</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={moduleData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="module" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [`${v}%`, 'High Risk']} contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                <Bar dataKey="high_risk_pct" radius={[4, 4, 0, 0]}>
                  {moduleData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  )
}
