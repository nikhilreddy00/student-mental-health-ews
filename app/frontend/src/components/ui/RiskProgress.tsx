interface Props {
  value: number
}

export default function RiskProgress({ value }: Props) {
  const pct = Math.round(value * 100)
  const color = value >= 0.66 ? 'bg-error' : value >= 0.40 ? 'bg-warning' : 'bg-success'
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-base-300 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] text-neutral-content tabular-nums">{value.toFixed(3)}</span>
    </div>
  )
}
