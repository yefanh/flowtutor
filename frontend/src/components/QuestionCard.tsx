import type { AnswerResult, HintResult, Question } from '../types'

const LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

interface QuestionCardProps {
  question: Question
  selected: number | null
  result: AnswerResult | null
  hint: HintResult | null
  hintPending: boolean
  onSelect: (index: number) => void
  onSubmit: () => void
  onAskForHint: () => void
  onNext: () => void
}

export default function QuestionCard({
  question,
  selected,
  result,
  hint,
  hintPending,
  onSelect,
  onSubmit,
  onAskForHint,
  onNext,
}: QuestionCardProps) {
  // Three states, not two. A wrong first answer is not the end of the
  // question -- the learner has another go, with a nudge if they want one.
  const settled = result?.revealed === true
  const retrying = result !== null && !result.revealed

  function optionClass(index: number): string {
    if (settled) {
      if (index === result.correct_answer) return 'option correct'
      if (index === selected) return 'option wrong'
      return 'option muted-option'
    }
    if (retrying) {
      // Their rejected choice stays marked; everything else is live again.
      // The component still has no idea which option is right, because the
      // server has not told it.
      if (index === selected) return 'option wrong'
      return 'option'
    }
    return selected === index ? 'option selected' : 'option'
  }

  return (
    <section className="card">
      <div className="meta">
        <span className="chip">{question.concept_name}</span>
        <span className="muted">Difficulty {question.difficulty}/5</span>
      </div>

      <h2 className="stem">{question.stem}</h2>

      <ul className="options">
        {question.options.map((option, index) => (
          <li key={index}>
            <button
              className={optionClass(index)}
              onClick={() => onSelect(index)}
              disabled={settled || (retrying && index === selected)}
            >
              <span className="label">{LABELS[index]}</span>
              <span>{option}</span>
            </button>
          </li>
        ))}
      </ul>

      {result === null && (
        <button className="primary" onClick={onSubmit} disabled={selected === null}>
          Submit
        </button>
      )}

      {retrying && (
        <div className="retry">
          <strong className="no">Not quite</strong>
          <p className="muted small">
            Have another go. The answer is still hidden — that is the point.
          </p>

          {hint && <HintPanel hint={hint} />}

          <div className="actions">
            <button className="primary" onClick={onSubmit} disabled={selected === null}>
              Try again
            </button>
            {!hint && (
              <button className="secondary" onClick={onAskForHint} disabled={hintPending}>
                {hintPending ? 'Thinking…' : 'Give me a hint'}
              </button>
            )}
          </div>
        </div>
      )}

      {settled && (
        <div className={`result ${result.is_correct ? 'ok' : 'no'}`}>
          <strong>{result.is_correct ? 'Correct' : 'Not quite'}</strong>
          {result.explanation && <p>{result.explanation}</p>}
          <MasteryChange result={result} />
          <button className="primary" onClick={onNext}>
            Next question
          </button>
        </div>
      )}
    </section>
  )
}

/**
 * The hint itself, and nothing else.
 *
 * An earlier version printed "Based on <top retrieved chunk>" underneath. It
 * was wrong often enough to matter: the top-ranked chunk is not necessarily
 * the one the hint drew on, so a hint saying "revisit lesson step 4" appeared
 * beneath a line crediting step 6. The hint already names what to reread, and
 * the full source list is recorded server-side, which is where it is actually
 * useful -- tracing a bad hint back to bad retrieval.
 */
function HintPanel({ hint }: { hint: HintResult }) {
  return (
    <div className="hint">
      <span className="hint-label">Hint</span>
      <p>{hint.hint}</p>
    </div>
  )
}

/**
 * Feedback tied to capability, not activity.
 *
 * The learner is told what changed about what they can do -- never how many
 * questions they have answered today or how many days they have kept it up.
 */
function MasteryChange({ result }: { result: AnswerResult }) {
  const { mastery } = result

  if (!result.mastery_updated) {
    return (
      <p className="muted small mastery-change">
        Already counted — one question only moves the estimate once.
      </p>
    )
  }

  const before = Math.round(mastery.previous * 100)
  const after = Math.round(mastery.current * 100)
  const rising = mastery.delta > 0

  return (
    <div className="mastery-change">
      {mastery.crossed_threshold && (
        <p className="breakthrough">You have {mastery.concept_name} solid now.</p>
      )}
      <p className="muted small">
        {mastery.concept_name} <span className="figure">{before}%</span>
        <span className={rising ? 'arrow up' : 'arrow down'}>
          {rising ? '↑' : '↓'}
        </span>
        <span className="figure">{after}%</span>
        {result.used_hint && rising && <span className="tag">with a hint</span>}
      </p>
    </div>
  )
}
