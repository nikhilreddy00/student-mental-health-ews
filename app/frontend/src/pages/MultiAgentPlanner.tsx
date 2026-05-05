import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, Lightbulb, Mail, Play, CheckCircle } from 'lucide-react'
import { getHighRiskIds, getStudent } from '../api/endpoints'
import { useSSE } from '../hooks/useSSE'
import PageHeader from '../components/layout/PageHeader'
import StatCard from '../components/ui/StatCard'
import RiskBadge from '../components/ui/RiskBadge'
import StreamingText from '../components/ui/StreamingText'

type Phase = 'idle' | 'start' | 'agent1' | 'agent2' | 'agent3' | 'done'

const AGENT_LABELS = [
  { key: 'agent1', title: 'Agent 1 — Risk Analyst', desc: 'Behavioral risk profile', Icon: Search },
  { key: 'agent2', title: 'Agent 2 — Intervention Advisor', desc: 'Ranked intervention strategies', Icon: Lightbulb },
  { key: 'agent3', title: 'Agent 3 — Outreach Writer', desc: 'Personalized student message', Icon: Mail },
]

export default function MultiAgentPlanner() {
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [texts, setTexts] = useState<Record<string, string>>({ agent1: '', agent2: '', agent3: '' })
  const [phase, setPhase] = useState<Phase>('idle')
  const { isStreaming, error, startStream } = useSSE()

  const { data: ids } = useQuery({
    queryKey: ['high-risk-ids'],
    queryFn: () => getHighRiskIds(false),
    staleTime: 30 * 60 * 1000,
  })

  const { data: student } = useQuery({
    queryKey: ['student', selectedId],
    queryFn: () => getStudent(selectedId!),
    enabled: selectedId !== null,
    staleTime: 10 * 60 * 1000,
  })

  async function runWorkflow() {
    if (!selectedId) return
    setTexts({ agent1: '', agent2: '', agent3: '' })
    setPhase('start')

    await startStream(
      `/api/multi-agent/${selectedId}`,
      {},
      (event) => {
        const { phase: p, payload } = event as { phase: string; payload: string }
        if (p === 'start') { setPhase('agent1') }
        else if (p === 'agent1') {
          setTexts(prev => ({ ...prev, agent1: prev.agent1 + payload }))
          setPhase('agent1')
        }
        else if (p === 'agent2') {
          setTexts(prev => ({ ...prev, agent2: prev.agent2 + payload }))
          setPhase('agent2')
        }
        else if (p === 'agent3') {
          setTexts(prev => ({ ...prev, agent3: prev.agent3 + payload }))
          setPhase('agent3')
        }
        else if (p === 'done') { setPhase('done') }
      },
    )
    setPhase('done')
  }

  return (
    <div>
      <PageHeader title="Multi-Agent Planner" subtitle="3-agent AI workflow: Risk → Interventions → Outreach" />
      <div className="p-6 space-y-6">

        {/* Student selector */}
        <div className="card bg-base-200 shadow-sm">
          <div className="card-body p-4">
            <div className="flex flex-wrap gap-3 items-end">
              <div className="form-control">
                <label className="label py-0 mb-1"><span className="label-text text-xs">Select High-Risk Student</span></label>
                <select
                  className="select select-sm select-bordered w-56"
                  value={selectedId ?? ''}
                  onChange={e => { setSelectedId(Number(e.target.value)); setTexts({ agent1: '', agent2: '', agent3: '' }); setPhase('idle') }}
                >
                  <option value="">-- Choose student --</option>
                  {ids?.ids.slice(0, 200).map(id => <option key={id} value={id}>{id}</option>)}
                </select>
              </div>
              <button
                className="btn btn-primary btn-sm"
                disabled={!selectedId || isStreaming}
                onClick={runWorkflow}
              >
                {isStreaming ? <><span className="loading loading-spinner loading-xs" /> Running agents…</> : <><Play size={13} /> Generate Intervention Plan</>}
              </button>
            </div>
          </div>
        </div>

        {/* Student snapshot */}
        {student && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Risk Score" value={student.risk_score.toFixed(3)} accent="high" mono />
            <div className="stat bg-base-200 rounded-box shadow-sm">
              <div className="stat-title text-xs uppercase opacity-70">Tier</div>
              <div className="stat-value text-xl mt-1"><RiskBadge tier={student.risk_tier} size="md" /></div>
            </div>
            <StatCard label="Module" value={student.code_module} />
            <StatCard label="Mean Score" value={`${student.mean_score.toFixed(1)}/100`} />
          </div>
        )}

        {error && <div className="alert alert-error text-sm">{error}</div>}

        {/* Agent output boxes */}
        {AGENT_LABELS.map(({ key, title, desc, Icon }) => {
          const text = texts[key]
          const isCurrentPhase = phase === key
          const hasDone = phase === 'done' || (key === 'agent1' && ['agent2', 'agent3', 'done'].includes(phase))
            || (key === 'agent2' && ['agent3', 'done'].includes(phase))

          if (!text && phase === 'idle') return null

          return (
            <div key={key} className={`card bg-base-200 shadow-sm border ${isCurrentPhase ? 'border-primary' : 'border-transparent'}`}>
              <div className="card-body p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon size={14} className="text-primary shrink-0" />
                    <div>
                      <h3 className="card-title text-sm">{title}</h3>
                      <p className="text-xs opacity-60">{desc}</p>
                    </div>
                  </div>
                  {hasDone && !isCurrentPhase && <span className="badge badge-success badge-sm">Complete</span>}
                  {isCurrentPhase && <span className="loading loading-dots loading-sm text-primary" />}
                </div>
                {text ? (
                  <StreamingText text={text} isStreaming={isCurrentPhase} className="mt-2" />
                ) : (
                  <div className="text-xs opacity-40 mt-2 italic">Waiting…</div>
                )}
              </div>
            </div>
          )
        })}

        {phase === 'done' && (
          <div className="alert alert-success flex items-center gap-2">
            <CheckCircle size={16} className="shrink-0" />
            Intervention plan complete for Student {selectedId}
          </div>
        )}

      </div>
    </div>
  )
}
