interface Props {
  title: string
  subtitle?: string
  action?: React.ReactNode
}

export default function PageHeader({ title, subtitle, action }: Props) {
  return (
    <div className="px-6 py-5 border-b border-base-300 flex items-center justify-between">
      <div className="flex items-start gap-3">
        <div className="w-0.5 self-stretch bg-primary rounded-full mt-0.5 shrink-0" />
        <div>
          <h1 className="text-base font-semibold text-base-content tracking-tight">{title}</h1>
          {subtitle && (
            <p className="text-xs text-neutral-content mt-0.5">{subtitle}</p>
          )}
        </div>
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}
