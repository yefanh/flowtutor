import type { LessonStep } from '../types'

interface LessonCardProps {
  lesson: LessonStep
  onContinue: () => void
  busy: boolean
}

/**
 * Teaching mode.
 *
 * Deliberately plain: one idea, some prose, and a way forward. No score, no
 * timer, no progress reward. Reading is not an achievement here -- it is the
 * groundwork that makes the practice questions worth answering.
 *
 * The step counter is the one exception, and it is there for orientation
 * ("how much more of this?"), not as something to complete for its own sake.
 */
export default function LessonCard({
  lesson,
  onContinue,
  busy,
}: LessonCardProps) {
  const last = lesson.step === lesson.total_steps

  return (
    <section className="card lesson">
      <div className="meta">
        <span className="chip chip-teach">{lesson.concept_name}</span>
        <span className="muted">
          Step {lesson.step} of {lesson.total_steps}
        </span>
      </div>

      <h2 className="lesson-title">{lesson.title}</h2>

      <LessonBody body={lesson.body} />

      <button className="primary" onClick={onContinue} disabled={busy}>
        {last ? 'Start practising' : 'Got it, next'}
      </button>
    </section>
  )
}

/**
 * Renders the lesson text.
 *
 * The body is plain text with two conventions: blank lines separate
 * paragraphs, and lines beginning with `*` are bullets. That is all the
 * structure the content needs, so it is all this parses -- pulling in a
 * markdown library to support syntax nothing uses would be dead weight.
 */
function LessonBody({ body }: { body: string }) {
  const blocks = body.trim().split(/\n\s*\n/)

  return (
    <div className="lesson-body">
      {blocks.map((block, index) => {
        const lines = block.split('\n').map((line) => line.trim())

        if (lines.every((line) => line.startsWith('*'))) {
          return (
            <ul key={index} className="lesson-bullets">
              {lines.map((line, i) => (
                <li key={i}>{line.replace(/^\*\s*/, '')}</li>
              ))}
            </ul>
          )
        }

        return <p key={index}>{lines.join(' ')}</p>
      })}
    </div>
  )
}
