import { useCallback, useEffect, useState } from 'react'

import {
  completeLessonStep,
  fetchMastery,
  fetchNext,
  submitAnswer,
} from './api'
import LessonCard from './components/LessonCard'
import MasteryPanel from './components/MasteryPanel'
import QuestionCard from './components/QuestionCard'
import type { AnswerResult, ConceptMastery, NextItem } from './types'

// No auth yet. A real user id arrives with accounts; until then every session
// is the same learner, which is enough to exercise the whole loop.
const USER_ID = 1

export default function App() {
  const [next, setNext] = useState<NextItem | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [result, setResult] = useState<AnswerResult | null>(null)
  const [mastery, setMastery] = useState<ConceptMastery[]>([])
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const message = (err: unknown) =>
    err instanceof Error ? err.message : String(err)

  const loadMastery = useCallback(async () => {
    try {
      setMastery(await fetchMastery(USER_ID))
    } catch (err) {
      setError(message(err))
    }
  }, [])

  const loadNext = useCallback(async () => {
    setLoading(true)
    setError(null)
    setSelected(null)
    setResult(null)
    try {
      const item = await fetchNext(USER_ID)
      setNext(item)
      // Time-on-task feeds the adaptive engine: a correct answer that took two
      // minutes is weaker evidence of mastery than an instant one.
      setStartedAt(Date.now())
    } catch (err) {
      setError(message(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadNext()
    loadMastery()
  }, [loadNext, loadMastery])

  async function handleLessonContinue() {
    if (next?.kind !== 'lesson') return
    setBusy(true)
    try {
      await completeLessonStep(USER_ID, next.lesson.lesson_id)
      await Promise.all([loadNext(), loadMastery()])
    } catch (err) {
      setError(message(err))
    } finally {
      setBusy(false)
    }
  }

  async function handleSubmit() {
    if (next?.kind !== 'question' || selected === null || result !== null) return
    try {
      const res = await submitAnswer({
        userId: USER_ID,
        questionId: next.question.id,
        selected,
        timeSpent: startedAt
          ? Math.round((Date.now() - startedAt) / 1000)
          : undefined,
      })
      setResult(res)
      // The answer just changed the estimate, so the progress view is stale.
      await loadMastery()
    } catch (err) {
      setError(message(err))
    }
  }

  const activeConceptId =
    next?.kind === 'lesson'
      ? next.lesson.concept_id
      : next?.kind === 'question'
        ? next.question.concept_id
        : undefined

  return (
    <div className="page">
      <header className="header">
        <h1>FlowTutor</h1>
        <p className="tagline">
          Learn it first, then practise at the edge of what you know.
        </p>
      </header>

      <main>
        {error && (
          <div className="banner banner-error">
            {error}
            <button className="link" onClick={loadNext}>
              retry
            </button>
          </div>
        )}

        {loading && <p className="muted">Loading…</p>}

        {!loading && next?.kind === 'lesson' && (
          <LessonCard
            lesson={next.lesson}
            onContinue={handleLessonContinue}
            busy={busy}
          />
        )}

        {!loading && next?.kind === 'question' && (
          <QuestionCard
            question={next.question}
            selected={selected}
            result={result}
            onSelect={setSelected}
            onSubmit={handleSubmit}
            onNext={loadNext}
          />
        )}

        {mastery.length > 0 && (
          <MasteryPanel mastery={mastery} highlightConceptId={activeConceptId} />
        )}
      </main>
    </div>
  )
}
