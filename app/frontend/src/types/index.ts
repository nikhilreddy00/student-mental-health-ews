// ── Shared data types ──────────────────────────────────────────────────────────

export interface TierCount {
  tier: string
  count: number
}

export interface HistogramBin {
  bin_start: number
  bin_end: number
  count: number
}

export interface ReasonCount {
  reason: string
  count: number
}

export interface ModuleBreakdown {
  module: string
  total: number
  high_risk: number
  high_risk_pct: number
  avg_score: number
}

export interface OverviewData {
  total: number
  high: number
  medium: number
  low: number
  avg_risk_score: number
  high_pct: number
  medium_pct: number
  low_pct: number
  tier_distribution: TierCount[]
  risk_histogram: Record<string, HistogramBin[]>
  top_reasons: ReasonCount[]
  module_breakdown: ModuleBreakdown[]
}

export interface AlertStudent {
  id_student: number
  risk_score: number
  risk_tier: string
  top_reason: string
  reason_2: string
  reason_3: string
  code_module: string
}

export interface BehavioralComparison {
  features: string[]
  high_risk_avgs: number[]
  low_risk_avgs: number[]
}

export interface AlertsData {
  total: number
  page: number
  page_size: number
  students: AlertStudent[]
  risk_factors: { factor: string; count: number }[]
  behavioral_comparison: BehavioralComparison
}

export interface ShapWaterfallItem {
  feature: string
  shap_value: number
  direction: 'positive' | 'negative'
}

export interface StudentProfile {
  student_id: number
  code_module: string
  risk_score: number
  risk_tier: string
  gender: string
  age_band: string
  region: string
  imd_band: string
  highest_education: string
  disability: string
  num_of_prev_attempts: number
  studied_credits: number
  top_reason: string
  reason_2: string
  reason_3: string
  engagement_span: number
  mean_score: number
  active_days: number
  submission_rate: number
  engagement_decline: number
  dropout_modules: number
  shap_waterfall: ShapWaterfallItem[]
  suggested_action: { tier: string; message: string }
}

export interface ModelMetrics {
  auc_roc: number
  f1: number
  precision: number
  recall: number
  accuracy: number
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface ModelPerformanceData {
  metrics: ModelMetrics
  confusion_matrix: number[][]
  confusion_matrix_pct: number[][]
  cv_score: number
  cv_std: number
  feature_importances: FeatureImportance[]
  labels: { rows: string[]; cols: string[] }
}

export interface ShapPlot {
  id: string
  title: string
  url: string
  exists: boolean
}

export interface ShapData {
  plots: ShapPlot[]
}

export interface DemographicGroup {
  group: string
  count: number
  avg_risk_score: number
  high_risk_flag_rate_pct: number
}

export interface FairnessData {
  gender: DemographicGroup[]
  age_band: DemographicGroup[]
  imd_band: DemographicGroup[]
  gender_histogram: Record<string, { bin_start: number; count: number }[]>
  age_band_avg_scores: { age_band: string; avg_risk_score: number }[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResponse {
  response: string
  tool_calls: { tool: string; input: Record<string, unknown> }[]
}

export interface MultiAgentEvent {
  phase: 'start' | 'agent1' | 'agent2' | 'agent3' | 'done' | 'ping' | 'error'
  payload: string | Record<string, unknown>
}

export interface Trajectory {
  student_id: number
  weeks: number[]
  risk_scores: number[]
  final_risk: number
  top_reason: string
  module: string
}

export interface RiskMonitorData {
  students: Trajectory[]
  max_week: number
  threshold: number
}

export interface Booking {
  booking_id: string
  student_id: string
  risk_score: number
  advisor_type: string
  date: string
  time_slot: string
  urgency: string
  status: string
  booked_at: string
}

export interface BookingsData {
  total: number
  immediate: number
  soon: number
  routine: number
  scheduled: number
  bookings: Booking[]
  advisor_types: string[]
  time_slots: string[]
}

export interface TemporalStudent {
  student_id: number
  module: string
  top_reason: string
  final_risk: number
  early_signal_week: number | null
  actual_alert_week: number | null
  delay_weeks: number | null
}

export interface TemporalMetrics {
  mean_delay: number
  median_delay: number
  max_delay: number
  n_total: number
  n_with_delay: number
}

export interface TemporalAnalysisData {
  metrics: TemporalMetrics
  students: TemporalStudent[]
  trajectories: Trajectory[]
}

export interface MetadataResponse {
  modules: string[]
  advisor_types: string[]
  time_slots: string[]
  risk_tiers: string[]
}

export interface KBDocument {
  title: string
  filename: string
  chunk_count: number
  preview: string
}
