"""The adaptive engine's core model: pure functions, no I/O.

WHAT THIS IS
    An online item-response model. Every question has a difficulty, every
    learner has an ability, and the model predicts the probability that this
    learner answers this question correctly. After the answer arrives, the
    ability estimate moves by the *prediction error* -- how surprising the
    outcome was. This is the same update rule as Elo ratings, and it is the
    online (stochastic-gradient) form of a 1-parameter IRT model.

WHY NOT "correct: +0.1, wrong: -0.1"
    A fixed step ignores the question. Getting an easy question right when you
    are already strong is no evidence of anything, and should barely move the
    estimate; getting a hard question right when you are weak is strong
    evidence and should move it a lot. Prediction error gives exactly that,
    for free, with no special cases:

        delta = learning_rate * (actual_outcome - predicted_probability)

    Expected outcome  -> error near zero -> estimate barely moves.
    Surprising outcome -> error near one  -> estimate moves hard.

WHY THE GUESSING FLOOR
    These are 4-option multiple choice questions, so a learner who knows
    nothing still scores 25%. Without accounting for that, the model reads a
    lucky guess on a difficulty-5 question as proof of mastery. The floor makes
    predicted probability bottom out at 0.25, which shrinks the credit given
    for succeeding on questions far above the learner's level.

WHY THE LEARNING RATE DECAYS
    The first answer in a concept is most of what we know about the learner, so
    it should move the estimate a lot. The fiftieth should not -- by then the
    estimate rests on real evidence and single answers are noise. A decaying
    rate is the cheap version of "uncertainty shrinks as evidence accumulates",
    which a full Bayesian model would track explicitly.

Every constant below is a tuning knob, and none of them is validated against
real learners yet. They are chosen to produce sensible behaviour, not fitted.
"""

import math
from dataclasses import dataclass

# --------------------------------------------------------------- calibration

DEFAULT_MASTERY = 0.3
"""Where a learner starts on an unseen concept. Deliberately below the middle:
assume some gap, let evidence raise it."""

MIN_MASTERY = 0.02
MAX_MASTERY = 0.98
"""Never let the estimate reach 0 or 1. A saturated estimate stops responding
to evidence, and no learner is ever certainly perfect or certainly hopeless."""

ABILITY_SCALE = 6.0
"""Maps mastery 0..1 onto ability -3..+3 (log-odds units)."""

DIFFICULTY_CENTER = 3.0
DIFFICULTY_SCALE = 0.75
"""Maps difficulty 1..5 onto -1.5..+1.5, the same units as ability. Difficulty
3 is the neutral rung: a learner at mastery 0.5 is an even match for it."""

GUESS_FLOOR = 0.25
"""One in four options. See "why the guessing floor" above."""

BASE_LEARNING_RATE = 0.30
EVIDENCE_HALFLIFE = 8.0
"""After 8 attempts on a concept the learning rate is halved, after 24 it is a
quarter. Early answers dominate; later ones refine."""

TARGET_SUCCESS = 0.75
"""The success rate the question selector aims for.

This is the flow zone expressed as a number. Assessment systems target 0.5,
because a coin-flip question extracts the most information about ability --
but this is a learning product, not an exam. The training literature puts
optimal learning nearer 80% success; below roughly 70% learners stall out and
disengage. 0.75 sits in that band, and is measured *including* lucky guesses,
which is why it is not higher.
"""

SLOW_ANSWER_SECONDS = 120
SLOW_ANSWER_DAMPING = 0.7
"""A correct answer that took a long time is weaker evidence of mastery than an
instant one -- the learner reasoned it out rather than knowing it. Damping
applies only to gains, never to penalties: hesitating should not cost you.

CAVEAT: this threshold is a guess. It should be replaced by a per-question
percentile once there is enough response-time data to compute one."""

HINT_ASSISTED_CREDIT = 0.4
"""How much of the normal gain a correct answer earns after a hint.

Getting there with help is real progress and should count -- but it is not the
same evidence as getting there alone, and paying full credit for it would make
the fastest route to a high score "ask for a hint every time". Mastery is meant
to predict what the learner can do unaided, so an assisted success is weaker
evidence of exactly that.

Applied to gains only. A wrong answer after a hint is not punished extra: the
hint was offered, taking it should never cost anything.

0.4 is a judgement call, not a measurement. Nothing in the data says 0.4 rather
than 0.3 or 0.5; it is set where a hinted answer clearly counts for less than
an unaided one without being worthless.
"""

MASTERY_THRESHOLD = 0.8
"""Where a concept counts as "cracked". Used for progress display only -- the
model itself has no notion of finished."""


# ------------------------------------------------------------------ the model


def _sigmoid(x: float) -> float:
    # Guard against overflow on extreme inputs; math.exp(-800) raises.
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def ability(mastery: float) -> float:
    """Mastery (0..1) as a log-odds ability score."""
    return (mastery - 0.5) * ABILITY_SCALE


def item_difficulty(difficulty: int) -> float:
    """Question difficulty (1..5) in the same units as ability."""
    return (difficulty - DIFFICULTY_CENTER) * DIFFICULTY_SCALE


def probability_correct(mastery: float, difficulty: int) -> float:
    """P(this learner answers this question correctly).

    Bounded below by the guessing floor, so it ranges over 0.25..1.0.
    """
    skill_gap = ability(mastery) - item_difficulty(difficulty)
    return GUESS_FLOOR + (1.0 - GUESS_FLOOR) * _sigmoid(skill_gap)


def learning_rate(attempts: int) -> float:
    """How far a single answer is allowed to move the estimate."""
    return BASE_LEARNING_RATE / (1.0 + attempts / EVIDENCE_HALFLIFE)


def target_difficulty(mastery: float) -> float:
    """The difficulty at which this learner should succeed TARGET_SUCCESS of
    the time -- the centre of their flow zone.

    Derived by inverting probability_correct: solve for the difficulty that
    makes the predicted probability equal the target.
    """
    # Strip the guessing floor to recover the sigmoid's own output.
    sigmoid_target = (TARGET_SUCCESS - GUESS_FLOOR) / (1.0 - GUESS_FLOOR)
    # Invert the sigmoid to get the required ability-minus-difficulty gap.
    required_gap = math.log(sigmoid_target / (1.0 - sigmoid_target))
    difficulty_in_logits = ability(mastery) - required_gap
    return DIFFICULTY_CENTER + difficulty_in_logits / DIFFICULTY_SCALE


@dataclass(frozen=True)
class MasteryUpdate:
    """The result of one answer. Everything needed to explain the move."""

    previous: float
    updated: float
    delta: float
    predicted_probability: float
    applied_learning_rate: float
    crossed_threshold: bool

    @property
    def was_surprising(self) -> bool:
        """True when the outcome contradicted the model's expectation."""
        return abs(self.delta) > 0.05


def update(
    mastery: float,
    attempts: int,
    difficulty: int,
    is_correct: bool,
    time_spent: int | None = None,
    used_hint: bool = False,
) -> MasteryUpdate:
    """Fold one answer into the mastery estimate."""
    predicted = probability_correct(mastery, difficulty)
    outcome = 1.0 if is_correct else 0.0
    rate = learning_rate(attempts)

    delta = rate * (outcome - predicted)

    # Both dampers apply to gains only, and they compound: a slow, hinted
    # correct answer is the weakest kind of evidence there is.
    if is_correct:
        if time_spent is not None and time_spent > SLOW_ANSWER_SECONDS:
            delta *= SLOW_ANSWER_DAMPING
        if used_hint:
            delta *= HINT_ASSISTED_CREDIT

    updated = min(MAX_MASTERY, max(MIN_MASTERY, mastery + delta))

    return MasteryUpdate(
        previous=mastery,
        updated=updated,
        delta=updated - mastery,
        predicted_probability=predicted,
        applied_learning_rate=rate,
        crossed_threshold=mastery < MASTERY_THRESHOLD <= updated,
    )
