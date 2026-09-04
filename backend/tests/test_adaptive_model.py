"""Tests for the adaptive model.

These need no database and no server: the model is pure functions, which is
most of the reason it was written that way. The important test is the last
one -- a simulated learner, to check the estimator actually converges rather
than merely moving in the right direction one step at a time.
"""

import random

import pytest

from adaptive import model

# ---------------------------------------------------------- predicted success


def test_probability_rises_with_mastery():
    probabilities = [model.probability_correct(m, 3) for m in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert probabilities == sorted(probabilities)


def test_probability_falls_with_difficulty():
    probabilities = [model.probability_correct(0.5, d) for d in range(1, 6)]
    assert probabilities == sorted(probabilities, reverse=True)


def test_guessing_floor_is_respected():
    """A learner who knows nothing still has one option in four."""
    hopeless = model.probability_correct(model.MIN_MASTERY, 5)
    assert hopeless >= model.GUESS_FLOOR
    assert hopeless < model.GUESS_FLOOR + 0.05


def test_probability_never_leaves_zero_one():
    for mastery in (0.0, 0.5, 1.0):
        for difficulty in range(1, 6):
            p = model.probability_correct(mastery, difficulty)
            assert 0.0 <= p <= 1.0


# ------------------------------------------------------------------- updating


def test_correct_raises_and_wrong_lowers():
    assert model.update(0.5, 0, 3, is_correct=True).delta > 0
    assert model.update(0.5, 0, 3, is_correct=False).delta < 0


def test_hard_win_is_worth_more_than_easy_win():
    """The whole reason for using prediction error instead of a fixed step."""
    easy = model.update(0.5, 0, 1, is_correct=True).delta
    hard = model.update(0.5, 0, 5, is_correct=True).delta
    assert hard > easy * 3


def test_easy_loss_costs_more_than_hard_loss():
    """Failing something you should have passed is the informative failure."""
    easy = model.update(0.5, 0, 1, is_correct=False).delta
    hard = model.update(0.5, 0, 5, is_correct=False).delta
    assert easy < hard  # both negative; easy is the larger drop


def test_learning_rate_decays_with_evidence():
    rates = [model.learning_rate(n) for n in (0, 8, 32, 128)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] == pytest.approx(model.BASE_LEARNING_RATE)


def test_later_answers_move_the_estimate_less():
    novice = model.update(0.5, 0, 3, is_correct=True).delta
    veteran = model.update(0.5, 100, 3, is_correct=True).delta
    assert veteran < novice / 5


def test_mastery_stays_inside_bounds():
    almost_perfect = model.update(model.MAX_MASTERY, 0, 5, is_correct=True)
    assert almost_perfect.updated <= model.MAX_MASTERY

    almost_hopeless = model.update(model.MIN_MASTERY, 0, 1, is_correct=False)
    assert almost_hopeless.updated >= model.MIN_MASTERY


def test_slow_correct_answers_earn_less():
    fast = model.update(0.5, 0, 3, is_correct=True, time_spent=5).delta
    slow = model.update(0.5, 0, 3, is_correct=True, time_spent=model.SLOW_ANSWER_SECONDS + 60).delta
    assert 0 < slow < fast


def test_slow_wrong_answers_are_not_punished_extra():
    """Hesitating should never cost more than answering wrong quickly."""
    fast = model.update(0.5, 0, 3, is_correct=False, time_spent=5).delta
    slow = model.update(
        0.5, 0, 3, is_correct=False, time_spent=model.SLOW_ANSWER_SECONDS + 60
    ).delta
    assert fast == slow


def test_threshold_crossing_is_reported_once():
    just_below = model.MASTERY_THRESHOLD - 0.01
    crossing = model.update(just_below, 0, 5, is_correct=True)
    assert crossing.crossed_threshold is True

    already_above = model.update(crossing.updated, 0, 5, is_correct=True)
    assert already_above.crossed_threshold is False


# ------------------------------------------------------------ target difficulty


def test_target_difficulty_rises_with_mastery():
    targets = [model.target_difficulty(m) for m in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert targets == sorted(targets)


def test_target_difficulty_hits_the_intended_success_rate():
    """The selector's promise: at the difficulty it targets, the learner is
    predicted to succeed TARGET_SUCCESS of the time."""
    for mastery in (0.3, 0.5, 0.7):
        target = model.target_difficulty(mastery)
        # probability_correct takes the rung directly; the target is continuous.
        predicted = model.probability_correct(mastery, target)
        assert predicted == pytest.approx(model.TARGET_SUCCESS, abs=1e-6)


# ----------------------------------------------------------------- simulation


def _simulate(true_mastery: float, rounds: int, seed: int) -> float:
    """Run the full estimate -> select -> answer -> update loop.

    The simulated learner answers correctly with the probability the model
    itself assigns to their TRUE mastery. That makes this a test of the
    estimator under its own generative assumptions -- a necessary condition
    for correctness, not evidence that the model describes real people.
    """
    rng = random.Random(seed)
    estimate = model.DEFAULT_MASTERY
    attempts = 0

    for _ in range(rounds):
        # The selector picks a rung; the bank only has 1..5.
        target = model.target_difficulty(estimate)
        difficulty = min(5, max(1, round(target)))

        true_probability = model.probability_correct(true_mastery, difficulty)
        is_correct = rng.random() < true_probability

        estimate = model.update(estimate, attempts, difficulty, is_correct).updated
        attempts += 1

    return estimate


@pytest.mark.parametrize("true_mastery", [0.25, 0.45, 0.65])
def test_estimate_converges_on_true_mastery(true_mastery):
    """Over many questions the estimate should find the learner's real level.

    Averaged over seeds because a single run is a random walk.
    """
    finals = [_simulate(true_mastery, rounds=300, seed=s) for s in range(12)]
    average = sum(finals) / len(finals)
    assert average == pytest.approx(true_mastery, abs=0.12)


def test_the_bank_only_resolves_a_band_of_ability():
    """KNOWN LIMITATION, asserted so it does not surprise anyone later.

    Five difficulty rungs can only distinguish learners whose target
    difficulty falls inside 1..5. Outside that band the selector clamps, every
    question served is one the learner is expected to pass (or fail), and the
    estimate drifts to the ceiling (or floor) instead of settling.

    Measured band: roughly 0.37 to 0.87. Widening it needs harder and easier
    CONTENT, not a better model -- worth knowing before blaming the maths.
    """
    assert model.target_difficulty(0.87) > 5.0
    assert model.target_difficulty(0.36) < 1.0

    # Inside the band, both ends stay resolvable.
    assert 1.0 < model.target_difficulty(0.40) < 5.0
    assert 1.0 < model.target_difficulty(0.85) < 5.0


def test_a_strong_learner_is_pushed_to_the_top_of_the_bank():
    """A learner above the ceiling should be recognised as strong, even though
    their exact level cannot be pinned down."""
    finals = [_simulate(0.95, rounds=300, seed=s) for s in range(6)]
    assert min(finals) > 0.85


def test_a_weak_learner_is_not_dragged_to_the_floor():
    """The mirror image: the bank bottoms out at difficulty 1, so a learner
    below the floor should sit near it rather than collapse to zero."""
    finals = [_simulate(0.1, rounds=300, seed=s) for s in range(6)]
    average = sum(finals) / len(finals)
    assert model.MIN_MASTERY < average < 0.35
