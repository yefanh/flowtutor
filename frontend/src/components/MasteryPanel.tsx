import type { ConceptMastery } from '../types'

interface MasteryPanelProps {
  mastery: ConceptMastery[]
  highlightConceptId?: number
}

/**
 * The progress surface.
 *
 * It shows what the learner can now do, and nothing about how often they show
 * up. There is deliberately no streak, no daily counter and no "don't break
 * the chain" device here: those reward the behaviour of opening the app rather
 * than the outcome of getting better, and they make leaving feel like failure.
 *
 * A concept reaching "solid" is a good place to stop, not a number to protect.
 */
export default function MasteryPanel({
  mastery,
  highlightConceptId,
}: MasteryPanelProps) {
  const solid = mastery.filter((c) => c.is_mastered).length

  return (
    <section className="mastery">
      <div className="mastery-head">
        <h3>Where you are</h3>
        <span className="muted">
          {solid} of {mastery.length} concepts solid
        </span>
      </div>

      <ul className="mastery-list">
        {mastery.map((concept) => (
          <Row
            key={concept.concept_id}
            concept={concept}
            active={concept.concept_id === highlightConceptId}
          />
        ))}
      </ul>
    </section>
  )
}

function Row({
  concept,
  active,
}: {
  concept: ConceptMastery
  active: boolean
}) {
  // A concept still being taught reports lesson progress instead of a mastery
  // percentage. Showing "12%" next to material the learner has not been shown
  // yet would be reporting a measurement nobody has taken -- the low score
  // means "untaught", not "bad at this".
  const teaching =
    concept.lesson_steps_total > 0 &&
    concept.lesson_steps_done < concept.lesson_steps_total

  // Every learner starts at an assumed 0.3, which is a prior and not a
  // measurement. Drawing that as a filled bar would claim the learner already
  // knows a third of something nobody has tested them on, so an untouched
  // concept shows an empty bar.
  const measured = concept.attempts > 0
  const percent = Math.round(concept.score * 100)
  const lessonPercent = concept.lesson_steps_total
    ? Math.round((concept.lesson_steps_done / concept.lesson_steps_total) * 100)
    : 0
  const barWidth = teaching ? lessonPercent : measured ? percent : 0

  return (
    <li className={active ? 'row active' : 'row'}>
      <div className="row-head">
        <span className="row-name">
          {concept.concept_name}
          {concept.is_mastered && <span className="solid-tag">solid</span>}
        </span>
        <span className="muted small">
          {teaching
            ? `learning ${concept.lesson_steps_done}/${concept.lesson_steps_total}`
            : concept.attempts === 0
              ? 'not started'
              : `${percent}%`}
        </span>
      </div>
      <div className="bar">
        <div
          className={
            teaching ? 'fill teaching' : concept.is_mastered ? 'fill done' : 'fill'
          }
          style={{ width: `${barWidth}%` }}
        />
      </div>
    </li>
  )
}
