// The API contract, mirroring the Pydantic response models in backend/main.py.
//
// Hand-written for now, because the contract is a handful of small shapes. Once
// Phase 2 adds hint and streaming endpoints, generate this file from the
// backend's own OpenAPI schema instead:
//
//     npx openapi-typescript http://localhost:8000/openapi.json -o src/types.ts
//
// At that point the two sides cannot drift apart silently -- renaming a field
// in FastAPI becomes a compile error here rather than an `undefined` on screen.

/** A question as the client is allowed to see it: no answer field, by design. */
export interface Question {
  id: number
  concept_id: number
  concept_name: string
  stem: string
  options: string[]
  difficulty: number
}

/** How one answer moved the learner's estimate for that concept. */
export interface MasteryDelta {
  concept_id: number
  concept_name: string
  previous: number
  current: number
  delta: number
  crossed_threshold: boolean
}

/** What comes back after submitting -- now the answer is fair game. */
export interface AnswerResult {
  is_correct: boolean
  correct_answer: number
  explanation: string | null
  attempt_id: number
  mastery: MasteryDelta
}

/** One row of the progress view. */
export interface ConceptMastery {
  concept_id: number
  concept_name: string
  score: number
  attempts: number
  is_mastered: boolean
}

export interface AnswerSubmission {
  userId: number
  questionId: number
  selected: number
  timeSpent?: number
}
