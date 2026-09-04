import { useCallback, useEffect, useState } from 'react'

import { fetchQuestion, submitAnswer } from './api'
import QuestionCard from './components/QuestionCard'
import type { AnswerResult, Question } from './types'

// No auth in Phase 0. A real user id arrives with accounts; until then every
// session is the same learner, which is enough to exercise the attempt log.
const USER_ID = 1

export default function App() {
  const [question, setQuestion] = useState<Question | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [result, setResult] = useState<AnswerResult | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadQuestion = useCallback(async () => {
    setLoading(true)
    setError(null)
    setSelected(null)
    setResult(null)
    try {
      const q = await fetchQuestion(USER_ID)
      setQuestion(q)
      // Time-on-task is a signal the adaptive engine uses in Phase 1: a correct
      // answer that took 90 seconds is weaker evidence than one that took 10.
      setStartedAt(Date.now())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadQuestion()
  }, [loadQuestion])

  async function handleSubmit() {
    if (selected === null || result !== null || question === null) return
    try {
      const res = await submitAnswer({
        userId: USER_ID,
        questionId: question.id,
        selected,
        timeSpent: startedAt ? Math.round((Date.now() - startedAt) / 1000) : undefined,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>FlowTutor</h1>
        <p className="tagline">Practice pitched just above your current level.</p>
      </header>

      <main>
        {error && (
          <div className="banner banner-error">
            {error}
            <button className="link" onClick={loadQuestion}>
              retry
            </button>
          </div>
        )}

        {loading && <p className="muted">Loading question…</p>}

        {!loading && question && (
          <QuestionCard
            question={question}
            selected={selected}
            result={result}
            onSelect={setSelected}
            onSubmit={handleSubmit}
            onNext={loadQuestion}
          />
        )}
      </main>
    </div>
  )
}
