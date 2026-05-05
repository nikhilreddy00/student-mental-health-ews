import { api } from './client'
import type {
  OverviewData, AlertsData, StudentProfile, ModelPerformanceData,
  ShapData, FairnessData, ChatMessage, ChatResponse, RiskMonitorData,
  BookingsData, TemporalAnalysisData, MetadataResponse, KBDocument,
} from '../types'

export const getMetadata = () =>
  api.get<MetadataResponse>('/metadata').then(r => r.data)

export const getOverview = (module: string, tiers: string[]) =>
  api.get<OverviewData>('/overview', { params: { module, tiers } }).then(r => r.data)

export const getAlerts = (params: {
  module: string; tiers: string[]; page: number; page_size: number
}) => api.get<AlertsData>('/alerts', { params }).then(r => r.data)

export const getHighRiskIds = (include_medium = false) =>
  api.get<{ ids: number[] }>('/high-risk-ids', { params: { include_medium } }).then(r => r.data)

export const searchStudents = (params: Record<string, string | number>) =>
  api.get('/students', { params }).then(r => r.data)

export const getStudent = (id: number) =>
  api.get<StudentProfile>(`/students/${id}`).then(r => r.data)

export const postNarrative = (id: number) =>
  api.post<{ narrative: string }>(`/students/${id}/narrative`).then(r => r.data)

export const getModelPerformance = () =>
  api.get<ModelPerformanceData>('/model-performance').then(r => r.data)

export const getShap = () =>
  api.get<ShapData>('/shap').then(r => r.data)

export const getFairness = (module: string, tiers: string[]) =>
  api.get<FairnessData>('/fairness', { params: { module, tiers } }).then(r => r.data)

export const postChat = (messages: ChatMessage[]) =>
  api.post<ChatResponse>('/chat', { messages }).then(r => r.data)

export const getRiskMonitor = () =>
  api.get<RiskMonitorData>('/risk-monitor').then(r => r.data)

export const getBookings = (params: Record<string, string>) =>
  api.get<BookingsData>('/bookings', { params }).then(r => r.data)

export const createBooking = (data: Record<string, unknown>) =>
  api.post('/bookings', data).then(r => r.data)

export const patchBooking = (id: string, status: string) =>
  api.patch(`/bookings/${id}`, { status }).then(r => r.data)

export const getBookingBriefing = (id: string) =>
  api.get(`/bookings/${id}/briefing`).then(r => r.data)

export const postKBChat = (query: string, history: ChatMessage[]) =>
  api.post('/kb-chat', { query, history }).then(r => r.data)

export const getKBDocuments = () =>
  api.get<{ documents: KBDocument[] }>('/kb-documents').then(r => r.data)

export const getTemporalAnalysis = (early_threshold: number, consecutive_weeks: number) =>
  api.get<TemporalAnalysisData>('/temporal-analysis', {
    params: { early_threshold, consecutive_weeks },
  }).then(r => r.data)
