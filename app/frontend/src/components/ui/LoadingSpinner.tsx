export default function LoadingSpinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-24">
      <div className="w-6 h-6 border-2 border-base-300 border-t-primary rounded-full animate-spin" />
      <p className="text-xs text-neutral-content font-mono">{label}</p>
    </div>
  )
}
