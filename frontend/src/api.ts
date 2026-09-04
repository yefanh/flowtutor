// Thin wrapper around the backend. All paths go through the Vite proxy (/api).

import type {
  AnswerResult,
  AnswerSubmission,
  ConceptMastery,
  Question,
} from './types'

const BASE = '/api'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export function fetchQuestion(userId: number): Promise<Question> {
  return request<Question>(`/question?user_id=${userId}`)
}

export function fetchMastery(userId: number): Promise<ConceptMastery[]> {
  return request<ConceptMastery[]>(`/mastery?user_id=${userId}`)
}

export function submitAnswer({
  userId,
  questionId,
  selected,
  timeSpent,
}: AnswerSubmission): Promise<AnswerResult> {
  return request<AnswerResult>('/answer', {
    method: 'POST',
    body: JSON.stringify({
      user_id: userId,
      question_id: questionId,
      selected,
      time_spent: timeSpent,
    }),
  })
}
