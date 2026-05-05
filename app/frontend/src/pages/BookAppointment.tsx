import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { CheckCircle, FileText } from 'lucide-react'
import { getHighRiskIds, getStudent, getMetadata, createBooking } from '../api/endpoints'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import RiskBadge from '../components/ui/RiskBadge'
import LoadingSpinner from '../components/ui/LoadingSpinner'

const URGENCY_BADGE: Record<string, string> = {
  Immediate: 'badge-error', Soon: 'badge-warning', Routine: 'badge-success',
}

function computeUrgency(score: number): string {
  if (score >= 0.80) return 'Immediate'
  if (score >= 0.66) return 'Soon'
  return 'Routine'
}

export default function BookAppointment() {
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [advisorType, setAdvisorType] = useState('')
  const [date, setDate] = useState('')
  const [timeSlot, setTimeSlot] = useState('')
  const [notes, setNotes] = useState('')
  const [result, setResult] = useState<{ booking_id: string; urgency: string; ai_briefing: string } | null>(null)

  const { data: ids } = useQuery({
    queryKey: ['high-risk-ids', true],
    queryFn: () => getHighRiskIds(true),
    staleTime: 30 * 60 * 1000,
  })

  const { data: meta } = useQuery({
    queryKey: ['metadata'],
    queryFn: getMetadata,
    staleTime: Infinity,
  })

  const { data: student, isLoading: studentLoading } = useQuery({
    queryKey: ['student', selectedId],
    queryFn: () => getStudent(selectedId!),
    enabled: selectedId !== null,
    staleTime: 10 * 60 * 1000,
  })

  const bookMutation = useMutation({
    mutationFn: () => createBooking({
      student_id: selectedId,
      advisor_type: advisorType,
      date,
      time_slot: timeSlot,
      notes,
      generate_briefing: true,
    }),
    onSuccess: (data) => {
      setResult(data)
    },
  })

  const urgency = student ? computeUrgency(student.risk_score) : null
  const canSubmit = selectedId && advisorType && date && timeSlot

  return (
    <div>
      <PageHeader title="Book Appointment" subtitle="Schedule risk-aware counselor appointments with AI advisor briefings" />
      <div className="p-6 space-y-6">

        {/* Student selector */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <h3 className="card-title text-sm mb-2">Select Student</h3>
            <select
              className="select select-bordered select-sm w-full max-w-xs"
              value={selectedId ?? ''}
              onChange={e => { setSelectedId(Number(e.target.value)); setResult(null) }}
            >
              <option value="">-- Select a student --</option>
              {ids?.ids.slice(0, 200).map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          </div>
        </div>

        {studentLoading && <LoadingSpinner label="Loading student data…" />}

        {student && (
          <>
            {/* Student snapshot */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Risk Score" value={student.risk_score.toFixed(3)} accent="high" mono />
              <div className="stat bg-base-200 rounded-box shadow-sm">
                <div className="stat-title text-xs uppercase opacity-70">Risk Tier</div>
                <div className="stat-value text-2xl mt-1"><RiskBadge tier={student.risk_tier} size="md" /></div>
              </div>
              <StatCard label="Module" value={student.code_module} />
              {urgency && (
                <div className="stat bg-base-200 rounded-box shadow-sm">
                  <div className="stat-title text-xs uppercase opacity-70">Urgency</div>
                  <div className="stat-value text-lg mt-1">
                    <span className={`badge ${URGENCY_BADGE[urgency]} badge-lg`}>{urgency}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Booking form */}
            <div className="card bg-base-200 shadow-sm">
              <div className="card-body p-4">
                <h3 className="card-title text-sm mb-3">Appointment Details</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="form-control">
                    <label className="label py-0 mb-1"><span className="label-text text-sm">Advisor Type</span></label>
                    <select
                      className="select select-bordered select-sm"
                      value={advisorType}
                      onChange={e => setAdvisorType(e.target.value)}
                    >
                      <option value="">-- Select advisor --</option>
                      {meta?.advisor_types.map(a => <option key={a}>{a}</option>)}
                    </select>
                  </div>

                  <div className="form-control">
                    <label className="label py-0 mb-1"><span className="label-text text-sm">Preferred Date</span></label>
                    <input
                      type="date"
                      className="input input-bordered input-sm"
                      value={date}
                      onChange={e => setDate(e.target.value)}
                      min={new Date().toISOString().split('T')[0]}
                    />
                  </div>

                  <div className="form-control">
                    <label className="label py-0 mb-1"><span className="label-text text-sm">Time Slot</span></label>
                    <select
                      className="select select-bordered select-sm"
                      value={timeSlot}
                      onChange={e => setTimeSlot(e.target.value)}
                    >
                      <option value="">-- Select time --</option>
                      {meta?.time_slots.map(t => <option key={t}>{t}</option>)}
                    </select>
                  </div>

                  <div className="form-control">
                    <label className="label py-0 mb-1"><span className="label-text text-sm">Notes (optional)</span></label>
                    <textarea
                      className="textarea textarea-bordered textarea-sm"
                      rows={2}
                      placeholder="Any context for the appointment…"
                      value={notes}
                      onChange={e => setNotes(e.target.value)}
                    />
                  </div>
                </div>

                <div className="mt-4">
                  <button
                    className="btn btn-primary"
                    disabled={!canSubmit || bookMutation.isPending}
                    onClick={() => bookMutation.mutate()}
                  >
                    {bookMutation.isPending
                      ? <><span className="loading loading-spinner loading-sm" /> Booking & generating briefing…</>
                      : 'Book Appointment'}
                  </button>
                </div>
              </div>
            </div>

            {/* Success result */}
            {result && (
              <div className="space-y-3">
                <div className="alert alert-success flex items-center gap-2">
                  <CheckCircle size={15} className="shrink-0" />
                  Booking confirmed! ID: <span className="font-mono font-bold">{result.booking_id}</span>
                  &nbsp;· Urgency: {result.urgency}
                </div>
                {result.ai_briefing && (
                  <details className="collapse collapse-arrow bg-base-200 shadow-sm">
                    <summary className="collapse-title font-semibold text-sm flex items-center gap-1.5"><FileText size={13} /> AI Advisor Briefing (Confidential)</summary>
                    <div className="collapse-content text-sm leading-relaxed pb-3">
                      {result.ai_briefing}
                    </div>
                  </details>
                )}
              </div>
            )}
          </>
        )}

      </div>
    </div>
  )
}
