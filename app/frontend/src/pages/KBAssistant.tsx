import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BookOpen } from 'lucide-react'
import { postKBChat, getKBDocuments } from '../api/endpoints'
import PageHeader from '../components/layout/PageHeader'
import type { ChatMessage } from '../types'

const SUGGESTED = [
  'How should I respond to a student expressing suicidal thoughts?',
  'What are signs of academic burnout?',
  'How do I make a first outreach call to a disengaged student?',
  'What referral pathways are available for mental health crises?',
  'How to support international students who may be struggling?',
  'What are trauma-informed care principles for academic settings?',
]

export default function KBAssistant() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [sources, setSources] = useState<Record<number, string[]>>({})
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showKB, setShowKB] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const { data: kbDocs } = useQuery({
    queryKey: ['kb-documents'],
    queryFn: getKBDocuments,
    staleTime: Infinity,
  })

  async function sendMessage(text: string) {
    if (!text.trim() || isLoading) return
    const userMsg: ChatMessage = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setHistory(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const result = await postKBChat(text, history)
      const assistantMsg: ChatMessage = { role: 'assistant', content: result.answer }
      const idx = messages.length + 1
      setMessages(prev => [...prev, assistantMsg])
      setHistory(prev => [...prev, assistantMsg])
      if (result.sources?.length) {
        setSources(prev => ({ ...prev, [idx]: result.sources }))
      }
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'KB service unavailable. Ensure the backend is running.' }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="KB Assistant" subtitle="RAG-powered counseling knowledge base — 12 institutional protocols" />
      <div className="flex-1 flex flex-col p-6 gap-4 overflow-hidden" style={{ maxHeight: 'calc(100vh - 120px)' }}>

        {/* Suggested */}
        {messages.length === 0 && (
          <div>
            <p className="text-xs opacity-60 uppercase tracking-widest mb-2">Suggested questions</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {SUGGESTED.map(q => (
                <button key={q} className="btn btn-sm btn-outline text-left justify-start h-auto py-2 normal-case text-xs" onClick={() => sendMessage(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-2">
          {messages.map((msg, i) => (
            <div key={i}>
              <div className={`chat ${msg.role === 'user' ? 'chat-end' : 'chat-start'}`}>
                <div className="chat-header opacity-50 text-xs mb-0.5">
                  {msg.role === 'user' ? 'Counselor' : 'KB Assistant'}
                </div>
                <div className={`chat-bubble ${msg.role === 'user' ? 'chat-bubble-primary' : 'chat-bubble-neutral'} text-sm whitespace-pre-wrap`}>
                  {msg.content}
                </div>
              </div>
              {sources[i] && (
                <div className="ml-16 mt-1">
                  <details className="collapse collapse-arrow bg-base-200 rounded-box text-xs">
                    <summary className="collapse-title py-1 px-3 min-h-0 flex items-center gap-1.5">
                      <BookOpen size={11} className="text-primary shrink-0" /> Sources: {sources[i].join(', ')}
                    </summary>
                    <div className="collapse-content px-3 pb-2">
                      <ul className="list-disc ml-4 mt-1 space-y-0.5">
                        {sources[i].map(s => <li key={s}>{s}</li>)}
                      </ul>
                    </div>
                  </details>
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
            placeholder="Ask a counseling guidance question…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input) }
            }}
          />
          <div className="flex flex-col gap-1">
            <button className="btn btn-primary btn-sm" onClick={() => sendMessage(input)} disabled={isLoading || !input.trim()}>Send</button>
            <button className="btn btn-ghost btn-sm" onClick={() => setShowKB(!showKB)}>Browse KB</button>
          </div>
        </div>

        {/* KB browser */}
        {showKB && kbDocs && (
          <div className="card bg-base-200 shadow-sm max-h-64 overflow-y-auto">
            <div className="card-body p-4">
              <h3 className="card-title text-sm">Knowledge Base Documents</h3>
              <div className="space-y-2">
                {kbDocs.documents.map(doc => (
                  <details key={doc.filename} className="collapse collapse-arrow bg-base-300 rounded-box">
                    <summary className="collapse-title text-sm py-2 min-h-0">{doc.title}</summary>
                    <div className="collapse-content text-xs opacity-70 pb-2">{doc.preview}…</div>
                  </details>
                ))}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
