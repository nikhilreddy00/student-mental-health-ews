import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { FileText } from 'lucide-react'
import { getBookings, patchBooking, getBookingBriefing } from '../api/endpoints'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import RiskProgress from '../components/ui/RiskProgress'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import ErrorAlert from '../components/ui/ErrorAlert'
import type { Booking } from '../types'

const URGENCY_BADGE: Record<string, string> = {
  Immediate: 'badge-error', Soon: 'badge-warning', Routine: 'badge-success',
}

export default function BookingQueue() {
  const queryClient = useQueryClient()
  const [statusFilter, setStatusFilter] = useState('All')
  const [advisorFilter, setAdvisorFilter] = useState('All')
  const [urgencyFilter, setUrgencyFilter] = useState('All')
  const [selectedBriefingId, setSelectedBriefingId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['bookings', statusFilter, advisorFilter, urgencyFilter],
    queryFn: () => getBookings({ status: statusFilter, advisor_type: advisorFilter, urgency: urgencyFilter }),
    staleTime: 0,
  })

  const { data: briefing } = useQuery({
    queryKey: ['briefing', selectedBriefingId],
    queryFn: () => getBookingBriefing(selectedBriefingId!),
    enabled: !!selectedBriefingId,
  })

  const patchMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => patchBooking(id, status),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bookings'] }),
  })

  if (isLoading) return <LoadingSpinner label="Loading booking queue…" />
  if (error || !data) return <ErrorAlert message="Failed to load bookings." />

  const advisorTypes = ['All', ...(data.advisor_types ?? [])]
  const urgencyLevels = ['All', 'Immediate', 'Soon', 'Routine']
  const statuses = ['All', 'Scheduled', 'Completed', 'Cancelled']

  return (
    <div>
      <PageHeader title="Booking Queue" subtitle="Manage all counselor appointment bookings" />
      <div className="p-6 space-y-6">

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard label="Total Bookings" value={data.total} accent="blue" mono />
          <StatCard label="Immediate" value={data.immediate} accent="high" mono />
          <StatCard label="Soon" value={data.soon} accent="med" mono />
          <StatCard label="Routine" value={data.routine} accent="low" mono />
          <StatCard label="Scheduled" value={data.scheduled} desc="Pending appointments" mono />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3">
          {([
            ['Status', statuses, statusFilter, setStatusFilter],
            ['Advisor Type', advisorTypes, advisorFilter, setAdvisorFilter],
            ['Urgency', urgencyLevels, urgencyFilter, setUrgencyFilter],
          ] as const).map(([label, opts, val, setter]) => (
            <div key={String(label)} className="form-control">
              <label className="label py-0 mb-1"><span className="label-text text-xs">{label}</span></label>
              <select
                className="select select-sm select-bordered"
                value={val}
                onChange={e => (setter as (v: string) => void)(e.target.value)}
              >
                {opts.map((o: string) => <option key={o}>{o}</option>)}
              </select>
            </div>
          ))}
        </div>

        {/* Table */}
        <div className="card bg-base-200 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="table table-sm text-sm">
              <thead className="bg-base-300 text-xs uppercase tracking-wide">
                <tr>
                  <th>Booking ID</th>
                  <th>Student</th>
                  <th>Risk</th>
                  <th>Advisor</th>
                  <th>Date / Time</th>
                  <th>Urgency</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.bookings.length === 0 && (
                  <tr><td colSpan={8} className="text-center py-10 opacity-40">No bookings found</td></tr>
                )}
                {data.bookings.map((b: Booking) => (
                  <tr key={b.booking_id} className="hover">
                    <td className="font-mono text-xs">{b.booking_id}</td>
                    <td className="font-mono font-semibold">{b.student_id}</td>
                    <td><RiskProgress value={Number(b.risk_score)} /></td>
                    <td className="text-xs">{b.advisor_type}</td>
                    <td className="text-xs">{b.date} {b.time_slot}</td>
                    <td><span className={`badge ${URGENCY_BADGE[b.urgency] ?? 'badge-ghost'} badge-sm`}>{b.urgency}</span></td>
                    <td><span className="badge badge-ghost badge-sm">{b.status}</span></td>
                    <td>
                      <div className="flex gap-1">
                        <select
                          className="select select-xs select-bordered"
                          defaultValue={b.status}
                          onChange={e => patchMutation.mutate({ id: b.booking_id, status: e.target.value })}
                        >
                          {['Scheduled', 'Completed', 'Cancelled'].map(s => <option key={s}>{s}</option>)}
                        </select>
                        <button
                          className="btn btn-xs btn-ghost"
                          onClick={() => setSelectedBriefingId(b.booking_id === selectedBriefingId ? null : b.booking_id)}
                        >
                          <FileText size={12} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Briefing viewer */}
        {briefing && selectedBriefingId && (
          <div className="card bg-base-200 shadow-sm border border-primary/20">
            <div className="card-body p-4">
              <div className="flex items-center justify-between">
                <h3 className="card-title text-sm">
                  Advisor Briefing — Student {briefing.student_id}
                  <span className={`badge ${URGENCY_BADGE[briefing.urgency] ?? 'badge-ghost'} badge-sm ml-2`}>{briefing.urgency}</span>
                </h3>
                <button className="btn btn-xs btn-ghost" onClick={() => setSelectedBriefingId(null)}>✕</button>
              </div>
              <div className="bg-base-300 rounded-box p-3 text-sm leading-relaxed mt-2">
                {briefing.ai_briefing || <span className="opacity-40">No briefing generated.</span>}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
