import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { getModelPerformance } from '../api/endpoints'
import { tooltipStyle } from '../lib/chartStyles'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'

const CELL_STYLES = [
  'bg-success/20 text-success',
  'bg-warning/20 text-warning',
  'bg-warning/20 text-warning',
  'bg-success/20 text-success',
]

export default function ModelPerformance() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['model-performance'],
    queryFn: getModelPerformance,
    staleTime: 30 * 60 * 1000,
  })

  if (isLoading) return <LoadingSpinner label="Loading model metrics…" />
  if (error || !data) return <ErrorAlert message="Failed to load model performance data." />

  const cm = data.confusion_matrix.flat()
  const cmPct = data.confusion_matrix_pct.flat()
  const rowLabels = data.labels.rows
  const colLabels = data.labels.cols

  return (
    <div>
      <PageHeader title="Model Performance" subtitle="XGBoost classifier evaluation — AUC 0.975" />
      <div className="p-6 space-y-6">

        {/* KPI metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard label="AUC-ROC" value={data.metrics.auc_roc.toFixed(3)} accent="blue" desc="Target ≥ 0.80" mono />
          <StatCard label="Accuracy" value={`${(data.metrics.accuracy * 100).toFixed(1)}%`} accent="low" mono />
          <StatCard label="F1 Score" value={data.metrics.f1.toFixed(3)} accent="blue" mono />
          <StatCard label="Precision" value={data.metrics.precision.toFixed(3)} desc="Low false positives" mono />
          <StatCard label="Recall" value={data.metrics.recall.toFixed(3)} desc="Catches 88.7% at-risk" mono />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Confusion matrix */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body">
              <h3 className="card-title text-sm">Confusion Matrix</h3>
              <div className="overflow-x-auto">
                <table className="table table-sm text-center text-xs">
                  <thead>
                    <tr>
                      <th />
                      {colLabels.map(c => <th key={c} className="text-xs font-semibold">{c}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {rowLabels.map((r, ri) => (
                      <tr key={r}>
                        <td className="font-semibold text-left text-xs">{r}</td>
                        {colLabels.map((_, ci) => {
                          const idx = ri * 2 + ci
                          return (
                            <td key={ci} className={`${CELL_STYLES[idx]} rounded p-3 font-bold`}>
                              <div className="text-lg">{cm[idx].toLocaleString()}</div>
                              <div className="text-xs font-normal opacity-70">{cmPct[idx]}%</div>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-2 text-xs opacity-60">
                CV AUC: {data.cv_score.toFixed(4)} ± {data.cv_std.toFixed(4)}
              </div>
            </div>
          </div>

          {/* Feature importances */}
          <div className="card bg-base-200 shadow-sm">
            <div className="card-body">
              <h3 className="card-title text-sm">Top Feature Importances (XGBoost)</h3>
              {data.feature_importances.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart
                    layout="vertical"
                    data={data.feature_importances.slice(0, 12)}
                    margin={{ top: 0, right: 16, left: 100, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                    <XAxis type="number" tick={{ fontSize: 10 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={100} />
                    <Tooltip formatter={(v: number) => [v.toFixed(4), 'Importance']} contentStyle={tooltipStyle.contentStyle} labelStyle={tooltipStyle.labelStyle} itemStyle={tooltipStyle.itemStyle} cursor={tooltipStyle.barCursor} />
                    <Bar dataKey="importance" fill="#6366f1" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="alert alert-info text-sm">Model file not loaded; importances unavailable.</div>
              )}
            </div>
          </div>
        </div>

        {/* Interpretation */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body">
            <h3 className="card-title text-sm">Metric Interpretation Guide</h3>
            <div className="overflow-x-auto">
              <table className="table table-sm table-zebra text-sm">
                <thead><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr></thead>
                <tbody>
                  <tr><td>AUC-ROC 0.975</td><td><span className="badge badge-success">Excellent</span></td><td>Model ranks at-risk students far above safe students</td></tr>
                  <tr><td>Precision 93.6%</td><td><span className="badge badge-info">High</span></td><td>93.6% of flagged students are genuinely at risk — low counselor alarm fatigue</td></tr>
                  <tr><td>Recall 88.7%</td><td><span className="badge badge-info">High</span></td><td>88.7% of at-risk students are caught — few missed cases</td></tr>
                  <tr><td>F1 0.911</td><td><span className="badge badge-success">Excellent</span></td><td>Strong balance between precision and recall</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
