import { useQuery } from '@tanstack/react-query'
import { getShap } from '../api/endpoints'
import PageHeader from '../components/layout/PageHeader'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'

export default function ShapExplainability() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['shap'],
    queryFn: getShap,
    staleTime: 30 * 60 * 1000,
  })

  if (isLoading) return <LoadingSpinner label="Loading SHAP plots…" />
  if (error || !data) return <ErrorAlert message="Failed to load SHAP data." />

  return (
    <div>
      <PageHeader title="SHAP Explainability" subtitle="Model transparency — what drives each student's risk score" />
      <div className="p-6 space-y-6">

        <div className="alert alert-info text-sm">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20A10 10 0 0012 2z" />
          </svg>
          <span>SHAP (SHapley Additive exPlanations) shows exactly which behavioral features push each student's risk score up or down. Red = increases risk, Blue = decreases risk.</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.plots.map(plot => (
            <div key={plot.id} className="card bg-base-200 shadow-sm">
              <div className="card-body p-4">
                <h3 className="card-title text-sm">{plot.title}</h3>
                {plot.exists ? (
                  <img
                    src={plot.url}
                    alt={plot.title}
                    className="w-full rounded-box mt-2"
                    loading="lazy"
                  />
                ) : (
                  <div className="alert alert-warning text-sm mt-2">
                    Plot not found at {plot.url}. Run the ML pipeline to generate SHAP plots.
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* How-to-read guide */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body">
            <h3 className="card-title text-sm">How to Read SHAP Plots</h3>
            <div className="overflow-x-auto">
              <table className="table table-sm table-zebra text-sm">
                <thead><tr><th>Plot</th><th>What it shows</th><th>How to use it</th></tr></thead>
                <tbody>
                  <tr><td>Beeswarm</td><td>Each dot = one student. Color = feature value (red = high). X position = SHAP impact.</td><td>Identify which features globally drive risk up vs down.</td></tr>
                  <tr><td>Bar Chart</td><td>Mean absolute SHAP per feature across all students.</td><td>Rank features by overall importance to the model.</td></tr>
                  <tr><td>Dependence</td><td>SHAP value vs. raw feature value for top features.</td><td>Understand non-linear relationships (e.g. threshold effects in engagement decline).</td></tr>
                  <tr><td>Individual</td><td>Waterfall for specific students showing their exact drivers.</td><td>Explain to counselors WHY a student was flagged.</td></tr>
                  <tr><td>Tier Heatmap</td><td>Average SHAP per feature broken down by High/Medium/Low tier.</td><td>Compare what makes High-risk vs Low-risk students different.</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
