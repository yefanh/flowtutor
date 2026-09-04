// Thin wrapper around the backend. All paths go through the Vite proxy (/api).

import type {
  AnswerResult,
  AnswerSubmission,
  ConceptMastery,
  NextItem,
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

/**
 * The single entry point for "what now".
 *
 * The client never asks "should I be taught or tested?" -- it asks what is
 * next and renders whichever kind comes back.
 */
export function fetchNext(userId: number): Promise<NextItem> {
  return request<NextItem>(`/next?user_id=${userId}`)
}

export function completeLessonStep(
  userId: number,
  lessonId: number,
): Promise<{ status: string }> {
  return request('/lesson/complete', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId, lesson_id: lessonId }),
  })
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
