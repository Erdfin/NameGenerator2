"""Trigram-based name generator.

Three generation modes:

``random``
    The classic approach the user asked for: sample character trigrams at
    random from a background English corpus and concatenate them, overlapping
    where the trigrams share characters.  The result is random-looking but
    built from real letter sequences, so it reads like a plausible word.

``markov``
    A trigram chain fitted to the *positive* examples in the dataset, so
    generation is biased toward names the user's data already likes.

``accepted``
    Generate in a loop and keep only the candidates the classifier passes.
    This is the mode where the random forest does real work rather than just
    filtering a list.

The background corpus defaults to the word list in :mod:`seed_data` but can be
replaced with any text file via ``--corpus``.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from seed_data import COMMON_WORDS

#: Boundary markers used to signal word start / word end in the trigram chain.
START = "^"
END = "$"

VOWELS = set("aeiou")
DEFAULT_MIN_LENGTH = 4
DEFAULT_MAX_LENGTH = 10
DEFAULT_VOWEL_RATIO = (0.28, 0.62)


def _read_corpus_words(path: str) -> List[str]:
    """Read a plain-text corpus into a list of lowercase alphabetic tokens."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read().lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if len(t) >= 2]


def _default_corpus() -> List[str]:
    return list(COMMON_WORDS)


class TrigramModel:
    """A character trigram chain over a word corpus.

    The chain is stored as a mapping from a 2-character context to a list of
    possible next characters, with word-boundary markers folded in so the model
    can start and end a name on its own.
    """

    def __init__(self, words: Iterable[str]) -> None:
        self.words = [w.lower() for w in words if w]
        self._chain: Dict[str, Counter] = defaultdict(Counter)
        self._contexts: List[str] = []
        self._build()

    def _build(self) -> None:
        for word in self.words:
            padded = START + word + END
            for i in range(len(padded) - 2):
                context = padded[i : i + 2]
                nxt = padded[i + 2]
                self._chain[context][nxt] += 1
        self._contexts = list(self._chain.keys())

    def __len__(self) -> int:
        return len(self._chain)

    @property
    def trigram_counts(self) -> Counter:
        """Counts of every observed 3-character window, used as features."""
        counts: Counter = Counter()
        for context, nxts in self._chain.items():
            for nxt, count in nxts.items():
                counts[context + nxt] += count
        return counts

    def sample(self, rng: random.Random, length: int) -> str:
        """Sample one name of roughly ``length`` characters from the chain."""
        if not self._chain:
            return ""
        context = START + self._rng_start(rng)
        out: List[str] = []
        for _ in range(max(1, length)):
            options = self._chain.get(context)
            if not options:
                break
            chars, weights = zip(*options.items())
            nxt = rng.choices(chars, weights=weights, k=1)[0]
            if nxt == END:
                break
            out.append(nxt)
            context = context[1] + nxt
        return "".join(out)

    def _rng_start(self, rng: random.Random) -> str:
        """Pick a plausible first letter for a new name."""
        starters = [
            c for c in "abcdefghijklmnopqrstuvwxyz"
            if (START + c) in self._chain
        ]
        if not starters:
            return "a"
        return rng.choice(starters)

    def score(self, text: str) -> float:
        """Average log-probability per character under the chain.

        Used as a feature: names that look like the corpus score higher.
        """
        counts = self.trigram_counts
        total = sum(counts.values())
        if total == 0:
            return 0.0
        padded = START + text.lower() + END
        total_log = 0.0
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            total_log += _log(counts.get(tri, 0) / total)
        return total_log / max(1, len(padded) - 2)


def _log(value: float) -> float:
    import math

    if value <= 0.0:
        return math.log(1e-9)
    return math.log(value)


class NameGenerator:
    """Generates candidate names using trigram concatenation.

    Args:
        corpus_words: words to build the background trigram model from.
        positives: names from the dataset that the user marked as *pass*.
            When present, a second trigram chain is fitted to them.
        seed: RNG seed for reproducible generation.
    """

    def __init__(
        self,
        corpus_words: Optional[Sequence[str]] = None,
        positives: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.corpus_words = list(corpus_words) if corpus_words else _default_corpus()
        self.corpus_model = TrigramModel(self.corpus_words)
        self.positive_model: Optional[TrigramModel] = None
        if positives:
            self.positive_model = TrigramModel(positives)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _vowel_ratio(self, name: str) -> float:
        if not name:
            return 0.0
        vowels = sum(1 for ch in name if ch in VOWELS)
        return vowels / len(name)

    def _looks_speakable(self, name: str) -> bool:
        if len(name) < 3:
            return False
        if name[0] not in VOWELS and name[0] not in "bcdfghjklmnpqrstvwxyz":
            return False
        low, high = DEFAULT_VOWEL_RATIO
        ratio = self._vowel_ratio(name)
        if ratio < low - 0.18 or ratio > high + 0.18:
            return False
        # Reject runs of 4+ identical or 4+ consonants in a row.
        run = 1
        for i in range(1, len(name)):
            if name[i] == name[i - 1]:
                run += 1
            else:
                run = 1
            if run >= 4:
                return False
        consonants = 0
        for ch in name:
            consonants = consonants + 1 if ch not in VOWELS else 0
            if consonants >= 4:
                return False
        return True

    def _random_trigram_name(self, length: int) -> str:
        """Concatenate randomly sampled trigrams into a name.

        The classic approach the user asked for: pick a random trigram from
        the corpus, then keep appending further trigrams.  Each new trigram is
        chosen to *continue* the one before it - it must start with the last
        two characters already emitted - so the result is a chain of real
        letter sequences rather than unrelated chunks jammed together.
        """
        model = self.corpus_model
        if not model:
            return ""

        # Start from a plausible opening trigram (one that begins a word).
        starts = [t for t in model.trigram_counts if t.startswith(START)]
        if not starts:
            starts = list(model.trigram_counts.keys())
        if not starts:
            return ""

        current = self.rng.choice(starts)
        out = [c for c in current if c not in (START, END)]

        while len(out) < length:
            # The next trigram must extend the current context, so the join is
            # always a real overlap rather than a hard seam.
            context = "".join(out[-2:]) if len(out) >= 2 else current[-2:]
            options = [
                t
                for t in model.trigram_counts
                if t.startswith(context) and not t.endswith(END)
            ]
            if not options:
                break
            weights = [model.trigram_counts[t] for t in options]
            nxt = self.rng.choices(options, weights=weights, k=1)[0]
            out.append(nxt[-1])

        return "".join(out)[:length]

    def _candidate(self, mode: str, length: int) -> str:
        if mode == "markov" and self.positive_model is not None:
            name = self.positive_model.sample(self.rng, length)
            if name:
                return name
        if mode == "random":
            return self._random_trigram_name(length)
        # "markov" without positives, and "accepted", start from the corpus.
        name = self.corpus_model.sample(self.rng, length)
        return name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        mode: str = "random",
        count: int = 1,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        attempts_multiplier: int = 60,
    ) -> List[str]:
        """Generate up to ``count`` unique candidate names.

        Args:
            mode: ``"random"``, ``"markov"`` or ``"accepted"``.
            count: how many names to return.
            min_length: minimum name length.
            max_length: maximum name length.
            attempts_multiplier: how many raw samples to try per requested
                name before giving up on the length/speakability constraints.

        Returns:
            A list of unique lowercase names.
        """
        if mode not in {"random", "markov", "accepted"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if max_length < min_length:
            min_length, max_length = max_length, min_length

        results: List[str] = []
        seen = set()
        max_attempts = max(1, count * attempts_multiplier)

        for _ in range(max_attempts):
            if len(results) >= count:
                break
            length = self.rng.randint(min_length, max_length)
            name = self._candidate(mode, length)
            name = name.strip().lower()
            if not name or not name.isalpha():
                continue
            if not self._looks_speakable(name):
                continue
            if name in seen:
                continue
            seen.add(name)
            results.append(name)
        return results

    def generate_until(
        self,
        scorer,
        mode: str = "random",
        wanted: int = 1,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        threshold: float = 0.5,
        max_attempts: int = 4000,
    ) -> List[Tuple[str, float]]:
        """Generate names until ``scorer`` accepts ``wanted`` of them.

        Args:
            scorer: callable taking a name and returning a float score.
            mode: generation mode to use.
            wanted: how many accepted names to collect.
            min_length: minimum name length.
            max_length: maximum name length.
            threshold: minimum score for a name to count as accepted.
            max_attempts: cap on total raw samples tried.

        Returns:
            List of ``(name, score)`` pairs that the scorer accepted.
        """
        accepted: List[Tuple[str, float]] = []
        seen = set()
        for _ in range(max_attempts):
            if len(accepted) >= wanted:
                break
            length = self.rng.randint(min_length, max_length)
            name = self._candidate(mode, length)
            name = name.strip().lower()
            if not name or not name.isalpha() or name in seen:
                continue
            if not self._looks_speakable(name):
                continue
            seen.add(name)
            score = float(scorer(name))
            if score >= threshold:
                accepted.append((name, score))
        return accepted


def load_corpus_from_file(path: str) -> List[str]:
    """Load corpus words from a text file (one word per line, or free text)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return _read_corpus_words(path)
