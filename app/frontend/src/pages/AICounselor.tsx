import { useState, useRef, useEffect } from 'react'
import { Wrench } from 'lucide-react'
import { postChat } from '../api/endpoints'
import PageHeader from '../components/layout/PageHeader'
import type { ChatMessage } from '../types'

const SUGGESTED = [
  'Show me the top 5 high-risk students',
  'Who are the most at-risk female students?',
  'Why is the highest-risk student flagged?',
  'Show high-risk students in module AAA',
  'What behavioral patterns indicate dropout risk?',
  'Find students with low engagement and poor scores',
]

export default function AICounselor() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [apiMessages, setApiMessages] = useState<ChatMessage[]>([])
  const [toolCalls, setToolCalls] = useState<Record<number, { tool: string; input: Record<string, unknown> }[]>>({})
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(text: string) {
    if (!text.trim() || isLoading) return
    const userMsg: ChatMessage = { role: 'user', content: text }
    const newDisplay = [...messages, userMsg]
    const newApi = [...apiMessages, userMsg]
    setMessages(newDisplay)
    setApiMessages(newApi)
    setInput('')
    setIsLoading(true)

    try {
      const result = await postChat(newApi)
      const assistantMsg: ChatMessage = { role: 'assistant', content: result.response }
      setMessages(prev => [...prev, assistantMsg])
      setApiMessages(prev => [...prev, assistantMsg])
      if (result.tool_calls?.length) {
        setToolCalls(prev => ({ ...prev, [newDisplay.length]: result.tool_calls }))
      }
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error contacting AI. Please check that the backend is running.' }])
    } finally {
      setIsLoading(false)
    }
  }

  function clearChat() {
    setMessages([])
    setApiMessages([])
    setToolCalls({})
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="AI Counselor" subtitle="Natural language assistant with tool use — ask about any student" />
      <div className="flex-1 flex flex-col p-6 gap-4 overflow-hidden" style={{ maxHeight: 'calc(100vh - 120px)' }}>

        {/* Suggested questions */}
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs opacity-60 uppercase tracking-widest">Try asking…</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED.map(q => (
                <button key={q} className="btn btn-sm btn-outline btn-primary" onClick={() => sendMessage(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat history */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-2">
          {messages.map((msg, i) => (
            <div key={i}>
              <div className={`chat ${msg.role === 'user' ? 'chat-end' : 'chat-start'}`}>
                <div className="chat-header opacity-50 text-xs mb-0.5">
                  {msg.role === 'user' ? 'Counselor' : 'AI Assistant'}
                </div>
                <div className={`chat-bubble ${msg.role === 'user' ? 'chat-bubble-primary' : 'chat-bubble-neutral'} text-sm whitespace-pre-wrap`}>
                  {msg.content}
                </div>
              </div>
              {/* Tool calls display */}
              {toolCalls[i] && (
                <div className="ml-16 mt-1 space-y-1">
                  {toolCalls[i].map((tc, j) => (
                    <details key={j} className="collapse collapse-arrow bg-base-200 rounded-box text-xs">
                      <summary className="collapse-title py-1 px-3 min-h-0 flex items-center gap-1.5">
                        <Wrench size={11} className="text-primary shrink-0" /> Tool: <span className="font-mono text-primary">{tc.tool}</span>
                      </summary>
                      <div className="collapse-content px-3 pb-2">
                        <pre className="text-xs overflow-auto bg-base-300 rounded p-2 mt-1">
                          {JSON.stringify(tc.input, null, 2)}
                        </pre>
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="chat chat-start">
              <div className="chat-bubble chat-bubble-neutral">
                <span className="loading loading-dots loading-sm" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="flex gap-2 items-end pt-2 border-t border-base-300">
          <textarea
            className="textarea textarea-bordered flex-1 text-sm resize-none"
            rows={2}
            placeholder="Ask about students, risk factors, interventions…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input) }
            }}
          />
          <div className="flex flex-col gap-1">
            <button className="btn btn-primary btn-sm" onClick={() => sendMessage(input)} disabled={isLoading || !input.trim()}>
              Send
            </button>
            <button className="btn btn-ghost btn-sm" onClick={clearChat}>Clear</button>
          </div>
        </div>

      </div>
    </div>
  )
}
