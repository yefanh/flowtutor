import type { AnswerResult, Question } from '../types'

const LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

interface QuestionCardProps {
  question: Question
  selected: number | null
  result: AnswerResult | null
  onSelect: (index: number) => void
  onSubmit: () => void
  onNext: () => void
}

export default function QuestionCard({
  question,
  selected,
  result,
  onSelect,
  onSubmit,
  onNext,
}: QuestionCardProps) {
  function optionClass(index: number): string {
    // Before submitting, the only state is "picked" or not -- the component has
    // no way to know what is correct, because the server has not told it.
    if (result === null) return selected === index ? 'option selected' : 'option'
    if (index === result.correct_answer) return 'option correct'
    if (index === selected) return 'option wrong'
    return 'option muted-option'
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
              disabled={result !== null}
            >
              <span className="label">{LABELS[index]}</span>
              <span>{option}</span>
            </button>
          </li>
        ))}
      </ul>

      {result === null ? (
        <button className="primary" onClick={onSubmit} disabled={selected === null}>
          Submit
        </button>
      ) : (
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
 * Feedback tied to capability, not activity.
 *
 * The learner is told what changed about what they can do -- never how many
 * questions they have answered today or how many days they have kept it up.
 */
function MasteryChange({ result }: { result: AnswerResult }) {
  const { mastery } = result
  const before = Math.round(mastery.previous * 100)
  const after = Math.round(mastery.current * 100)
  const rising = mastery.delta > 0

  return (
    <div className="mastery-change">
      {mastery.crossed_threshold && (
        <p className="breakthrough">
          You have {mastery.concept_name} solid now.
        </p>
      )}
      <p className="muted small">
        {mastery.concept_name}{' '}
        <span className="figure">{before}%</span>
        <span className={rising ? 'arrow up' : 'arrow down'}>
          {rising ? '↑' : '↓'}
        </span>
        <span className="figure">{after}%</span>
      </p>
    </div>
  )
}
