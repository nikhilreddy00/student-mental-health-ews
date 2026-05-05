interface Props {
  label: string
  value: string | number
  desc?: string
  accent?: 'high' | 'med' | 'low' | 'blue' | 'muted'
  mono?: boolean
}

export default function StatCard({ label, value, desc, accent = 'muted', mono = false }: Props) {
  return (
    <div className={`panel px-4 py-3.5 stat-accent-${accent}`}>
      <div className="text-[10px] font-semibold uppercase tracking-widest text-neutral-content mb-1.5">
        {label}
      </div>
      <div className={`text-2xl font-bold text-base-content leading-none ${mono ? 'font-mono data-value' : ''}`}>
        {value}
      </div>
      {desc && (
        <div className="text-[11px] text-neutral-content mt-1.5">{desc}</div>
      )}
    </div>
  )
}
