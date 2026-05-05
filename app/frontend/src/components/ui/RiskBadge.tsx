interface Props {
  tier: string
  size?: 'sm' | 'md'
}

const config: Record<string, { dot: string; text: string; bg: string }> = {
  High:   { dot: 'bg-error',   text: 'text-error',   bg: 'bg-error/10 border-error/20' },
  Medium: { dot: 'bg-warning', text: 'text-warning', bg: 'bg-warning/10 border-warning/20' },
  Low:    { dot: 'bg-success', text: 'text-success', bg: 'bg-success/10 border-success/20' },
}

export default function RiskBadge({ tier, size = 'md' }: Props) {
  const c = config[tier] ?? { dot: 'bg-neutral', text: 'text-neutral-content', bg: 'bg-base-300 border-base-300' }
  const px = size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-xs'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded border font-medium ${c.bg} ${c.text} ${px}`}>
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />
      {tier}
    </span>
  )
}
