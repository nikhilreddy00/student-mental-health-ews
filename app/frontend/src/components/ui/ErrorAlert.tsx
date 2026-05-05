import { AlertCircle } from 'lucide-react'

export default function ErrorAlert({ message }: { message: string }) {
  return (
    <div className="mx-6 my-4 flex items-start gap-3 panel border-error/30 px-4 py-3">
      <AlertCircle size={15} className="text-error shrink-0 mt-0.5" />
      <p className="text-sm text-base-content">{message}</p>
    </div>
  )
}
