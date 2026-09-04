"""Checking that a hint does not give the answer away.

WHY A PROGRAM AND NOT JUST A PROMPT
    The system prompt tells the model not to reveal the answer. That is a
    request. Requests are usually honoured and occasionally are not, and the
    one time it is not is a learner handed the answer to a question they were
    about to work out for themselves -- which is the entire product.

    So there are three defences, weakest last:

      1. The model is never told which option is correct. It cannot reveal what
         it does not know.
      2. Retrieval excludes the question's own explanation, which is a
         restatement of its answer.
      3. This module reads the generated hint and rejects it if the answer
         turns up anyway -- deduced from the reference material, or guessed.

    Only the third can fail open, and it is the one that catches the other two
    being wrong.

HOW IT DECIDES
    Two tests, either of which rejects:

      * a run of four consecutive content words from the answer appears in the
        hint -- quoting it, in other words
      * three quarters of the answer's distinctive words appear in the hint,
        in any order -- paraphrasing it
      * all but at most one of them appear, however few there are

    Both ignore stopwords and the words already in the question, because those
    are shared by every option and would fire on any hint about the topic.

    Both also compare stems rather than surface forms. Without that, "the
    application code fills the cache when it misses" slipped past a check that
    was looking for "miss" -- a paraphrase that gives away as much as a quote,
    defeated by a plural. Suffix rules are their own small science, so this
    uses a real stemmer rather than a hand-rolled one that is subtly wrong.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

NGRAM = 4
"""Length of a verbatim run that counts as quoting the answer."""

OVERLAP_THRESHOLD = 0.75
"""Fraction of the answer's distinctive words that counts as paraphrasing it."""

MAX_MISSING_WORDS = 1
"""How many of the answer's distinctive words may be absent before the ratio
rule stops applying.

A ratio alone is too coarse for short answers. "The application code, on a
cache miss" has three distinctive words once the question's own wording is
subtracted, so the ratio can only be 0, 33, 67 or 100 percent -- and a hint
reading "the cache is a passive box that your own application code must fill"
scored 67 and passed, having given away everything except the timing.

Found by using the thing, not by reasoning about it: the hint above came out of
the running app while clicking through a question by hand.
"""

MIN_COVERED_WORDS = 2
"""...but at least this many must be present, so a two-word answer is not
flagged the moment one of its words is mentioned."""

# Small and deliberate: a general-purpose stopword list would strip domain terms
# that carry meaning here. These are only the words that appear in every option
# regardless of topic.
STOPWORDS = frozenset(
    """
a an and are as at be been but by can do does for from had has have how if in
into is it its more most no not of on only or over should so than that the
their them then there these they this those to up use used uses using was what
when where which while who why will with would you your
""".split()
)


@dataclass(frozen=True)
class GuardrailResult:
    leaked: bool
    reason: str | None = None


@lru_cache(maxsize=1)
def _stemmer():
    return snowballstemmer.stemmer("english")


def _content_words(text: str, ignore: frozenset[str] = frozenset()) -> list[str]:
    """Distinctive words of a piece of text, stemmed.

    Stopwords go first, then anything already in the question, then stemming --
    so "misses" and "miss" are the same word by the time they are compared.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    kept = [w for w in words if w not in STOPWORDS]
    stemmed = _stemmer().stemWords(kept)
    return [w for w in stemmed if w not in ignore]


def check(hint: str, correct_option: str, question_stem: str = "") -> GuardrailResult:
    """Does this hint give away `correct_option`?

    `question_stem` is subtracted from the comparison: words already in the
    question are shared by every option, so counting them would flag any hint
    that stays on topic.
    """
    stem_words = frozenset(_content_words(question_stem))
    answer_words = _content_words(correct_option, ignore=stem_words)
    if not answer_words:
        # Nothing distinctive to protect -- an option made entirely of words
        # already in the question cannot be leaked by restating them.
        return GuardrailResult(leaked=False)

    hint_words = _content_words(hint, ignore=stem_words)
    hint_text = " ".join(hint_words)

    if len(answer_words) >= NGRAM:
        for i in range(len(answer_words) - NGRAM + 1):
            run = " ".join(answer_words[i : i + NGRAM])
            if run in hint_text:
                return GuardrailResult(True, f"quotes the answer: {run!r}")

    hint_set = set(hint_words)
    distinct = set(answer_words)
    covered = sum(1 for w in distinct if w in hint_set)
    missing = len(distinct) - covered
    ratio = covered / len(distinct)

    if ratio >= OVERLAP_THRESHOLD:
        return GuardrailResult(
            True, f"paraphrases the answer ({ratio:.0%} of its distinctive words)"
        )

    if covered >= MIN_COVERED_WORDS and missing <= MAX_MISSING_WORDS:
        return GuardrailResult(
            True,
            f"paraphrases the answer (all but {missing} of its {len(distinct)} distinctive words)",
        )

    return GuardrailResult(leaked=False)
