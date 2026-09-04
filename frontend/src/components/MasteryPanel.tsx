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
          <li
            key={concept.concept_id}
            className={concept.concept_id === highlightConceptId ? 'row active' : 'row'}
          >
            <div className="row-head">
              <span className="row-name">
                {concept.concept_name}
                {concept.is_mastered && <span className="solid-tag">solid</span>}
              </span>
              <span className="muted small">
                {concept.attempts === 0
                  ? 'not started'
                  : `${Math.round(concept.score * 100)}%`}
              </span>
            </div>
            <div className="bar">
              <div
                className={concept.is_mastered ? 'fill done' : 'fill'}
                style={{ width: `${Math.round(concept.score * 100)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
