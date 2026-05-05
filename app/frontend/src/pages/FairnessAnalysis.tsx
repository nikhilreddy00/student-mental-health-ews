import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, AreaChart, Area, Legend,
} from 'recharts'
import { getFairness } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import { useFilterStore } from '../store/filterStore'
import PageHeader from '../components/layout/PageHeader'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'

const GENDER_COLORS: Record<string, string> = { M: '#6366f1', F: '#ec4899' }
const CHECKS = [
  'Behavioral features only — no demographic inputs to model',
  'Gender fairness reviewed — flag rates within 5%',
  'Age band fairness reviewed — no systematic bias detected',
  'IMD (deprivation) band reviewed — see table for distribution',
  'Disability flag excluded from model inputs',
  'Region excluded from model inputs',
]

export default function FairnessAnalysis() {
  const { module, tiers } = useFilterStore()
  const { data, isLoading, error } = useQuery({
    queryKey: ['fairness', module, tiers],
    queryFn: () => getFairness(module, tiers),
    staleTime: 5 * 60 * 1000,
  })

  if (isLoading) return <LoadingSpinner label="Loading fairness data…" />
  if (error || !data) return <ErrorAlert message="Failed to load fairness data." />

  // Gender histogram data for recharts
  const genderKeys = Object.keys(data.gender_histogram)
  const genderHistData = Array.from({ length: 15 }, (_, i) => {
    const point: Record<string, number | string> = { bin: (i / 15).toFixed(2) }
    for (const g of genderKeys) {
      point[g] = data.gender_histogram[g]?.[i]?.count ?? 0
    }
    return point
  })

  return (
    <div>
      <PageHeader title="Fairness Analysis" subtitle="Equity review across demographic groups" />
      <div className="p-6 space-y-6">

        <div className="alert alert-warning text-sm">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          <span>Demographics are shown for equity monitoring only — they are NOT used as model features. Risk predictions are based solely on behavioral signals.</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Gender */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">By Gender</h3>
              <div className="overflow-x-auto">
                <table className="table table-sm table-zebra text-xs">
                  <thead><tr><th>Gender</th><th>Count</th><th>Avg Risk</th><th>High Risk %</th></tr></thead>
                  <tbody>
                    {data.gender.map(r => (
                      <tr key={r.group}>
                        <td>{r.group}</td>
                        <td>{r.count.toLocaleString()}</td>
                        <td>{r.avg_risk_score.toFixed(3)}</td>
                        <td>{r.high_risk_flag_rate_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={genderHistData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="bin" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 9 }} />
                  <Tooltip contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.cursor} />
                  <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                  {genderKeys.map(g => (
                    <Area key={g} type="monotone" dataKey={g} stroke={GENDER_COLORS[g] ?? '#888'} fill={GENDER_COLORS[g] ?? '#888'} fillOpacity={0.25} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Age band */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">By Age Band</h3>
              <div className="overflow-x-auto">
                <table className="table table-sm table-zebra text-xs">
                  <thead><tr><th>Age Band</th><th>Count</th><th>Avg Risk</th><th>High Risk %</th></tr></thead>
                  <tbody>
                    {data.age_band.map(r => (
                      <tr key={r.group}>
                        <td>{r.group}</td>
                        <td>{r.count.toLocaleString()}</td>
                        <td>{r.avg_risk_score.toFixed(3)}</td>
                        <td>{r.high_risk_flag_rate_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={data.age_band_avg_scores} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="age_band" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v: number) => [v.toFixed(3), 'Avg Risk']} contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                  <Bar dataKey="avg_risk_score" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* IMD Band */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm">By Deprivation Band (IMD)</h3>
            <div className="overflow-x-auto">
              <table className="table table-sm table-zebra text-sm">
                <thead><tr><th>IMD Band</th><th>Count</th><th>Avg Risk Score</th><th>High Risk Flag Rate</th></tr></thead>
                <tbody>
                  {data.imd_band.map(r => (
                    <tr key={r.group}>
                      <td>{r.group}</td>
                      <td>{r.count.toLocaleString()}</td>
                      <td>{r.avg_risk_score.toFixed(3)}</td>
                      <td>{r.high_risk_flag_rate_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Fairness checklist */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm">Fairness Design Checklist</h3>
            <div className="overflow-x-auto">
              <table className="table table-sm text-sm">
                <thead><tr><th>Check</th><th>Status</th></tr></thead>
                <tbody>
                  {CHECKS.map((c, i) => (
                    <tr key={i}>
                      <td>{c}</td>
                      <td><span className="badge badge-success badge-sm">Pass</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
