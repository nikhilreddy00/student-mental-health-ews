interface Props {
  text: string
  isStreaming?: boolean
  className?: string
}

export default function StreamingText({ text, isStreaming = false, className = '' }: Props) {
  return (
    <div className={`font-mono text-sm leading-relaxed whitespace-pre-wrap text-base-content ${className}`}>
      {text}
      {isStreaming && (
        <span className="streaming-cursor inline-block w-0.5 h-4 bg-primary ml-0.5 align-middle" />
      )}
    </div>
  )
}
