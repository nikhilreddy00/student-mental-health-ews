import { useState, useCallback } from 'react'
import { streamSSE } from '../api/streaming'

export function useSSE() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startStream = useCallback(
    async (
      url: string,
      body: Record<string, unknown>,
      onEvent: (e: Record<string, unknown>) => void,
    ) => {
      setIsStreaming(true)
      setError(null)
      try {
        for await (const event of streamSSE(url, body)) {
          onEvent(event)
          if (event.phase === 'done' || event.phase === 'error') break
          if ('done' in event && event.done) break
        }
      } catch (e) {
        setError(String(e))
      } finally {
        setIsStreaming(false)
      }
    },
    [],
  )

  return { isStreaming, error, startStream }
}
