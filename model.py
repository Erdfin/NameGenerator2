"""Random forest classifier that grades generated names as pass / fail.

Design notes
------------
**Training is a pure full refit.**  ``fit`` trains a single
:class:`~sklearn.ensemble.RandomForestClassifier` on the entire labelled
dataset.  There is no strategy search, no sample weighting, no recent-example
boosting.  The consequence is that training is a *pure function of the CSV
contents*: the same rows always produce the same model, byte for byte.

Determinism is enforced by:

* a fixed ``random_state`` on the forest,
* ``n_jobs=1`` so the tree-building and vote-averaging order cannot vary
  between runs,
* a fixed ``random_state`` on the cross-validation splitter used for the
  informational accuracy readout.

The cross-validated score that :meth:`NameClassifier.cross_val_score`
reports is *informational only*.  It never selects a model, never gates a
retrain, and never changes which model is active.  Deciding what to do with
that number is the user's call.

Features
--------
Each name is described by two feature blocks that are horizontally stacked:

*Hand-crafted structural features* (13 dense floats): length, vowel and
consonant ratios, longest vowel/consonant runs, repeated-letter counts,
distinct-letter count, bigram and trigram log-probability under a background
English model, leading/trailing vowel flags, and a vowel-consonant
alternation score.

*Character n-gram features*: a ``CountVectorizer`` over character 1-3 grams,
which lets the forest memorise letter sequences that look like the examples
the user marked as good.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from dataset import Dataset
from generator import END, START, VOWELS, TrigramModel
from seed_data import COMMON_WORDS

#: Fixed seed used everywhere, so retraining is reproducible.
RANDOM_STATE = 20240101

DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH: Optional[int] = None
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_CV_SPLITS = 5


class NameFeatures(BaseEstimator, TransformerMixin):
    """Turn a list of names into a feature matrix.

    Uses character n-grams (sparse) stacked with hand-crafted structural
    features (dense).
    """

    def __init__(
        self,
        ngram_min: int = 1,
        ngram_max: int = 3,
        min_df: int = 1,
    ) -> None:
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.min_df = min_df

    # -- sklearn API ----------------------------------------------------
    def fit(self, X, y=None):  # noqa: N803 (sklearn naming)
        self.vectorizer_ = CountVectorizer(
            analyzer="char",
            ngram_range=(self.ngram_min, self.ngram_max),
            min_df=self.min_df,
            lowercase=True,
        )
        self.vectorizer_.fit(list(X))
        self.background_ = TrigramModel(COMMON_WORDS)
        self.feature_names_out_ = self._feature_names()
        return self

    def transform(self, X):  # noqa: N803 (sklearn naming)
        names = [str(n).lower() for n in X]
        dense = np.asarray(
            [self._dense_features(name) for name in names], dtype=np.float64
        )
        char_features = self.vectorizer_.transform(names)
        return sparse.hstack(
            [sparse.csr_matrix(dense), char_features], format="csr"
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    # -- internals ------------------------------------------------------
    def _feature_names(self) -> List[str]:
        names = [
            "len",
            "vowel_ratio",
            "consonant_ratio",
            "max_vowel_run",
            "max_consonant_run",
            "repeated_letters",
            "distinct_letters",
            "bigram_logprob",
            "trigram_logprob",
            "starts_vowel",
            "ends_vowel",
            "alternation",
            "has_digit",
        ]
        try:
            names += [f"char_{f}" for f in self.vectorizer_.get_feature_names_out()]
        except Exception:  # pragma: no cover - defensive
            pass
        return names

    def _dense_features(self, name: str) -> List[float]:
        n = len(name)
        if n == 0:
            return [0.0] * 13

        vowels = sum(1 for ch in name if ch in VOWELS)
        vowel_ratio = vowels / n
        consonant_ratio = 1.0 - vowel_ratio

        max_vowel_run = _max_run(name, lambda ch: ch in VOWELS)
        max_consonant_run = _max_run(name, lambda ch: ch not in VOWELS)

        counts: Dict[str, int] = {}
        for ch in name:
            counts[ch] = counts.get(ch, 0) + 1
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        distinct = len(counts)

        bigram_lp = self._ngram_logprob(name, 2)
        trigram_lp = self._ngram_logprob(name, 3)

        starts_vowel = 1.0 if name[0] in VOWELS else 0.0
        ends_vowel = 1.0 if name[-1] in VOWELS else 0.0

        # Vowel/consonant alternation: fraction of adjacent pairs that switch.
        switches = 0
        for i in range(1, n):
            if (name[i] in VOWELS) != (name[i - 1] in VOWELS):
                switches += 1
        alternation = switches / (n - 1) if n > 1 else 0.0

        has_digit = 1.0 if any(ch.isdigit() for ch in name) else 0.0

        return [
            float(n),
            vowel_ratio,
            consonant_ratio,
            float(max_vowel_run),
            float(max_consonant_run),
            float(repeated),
            float(distinct),
            bigram_lp,
            trigram_lp,
            starts_vowel,
            ends_vowel,
            alternation,
            has_digit,
        ]

    def _ngram_logprob(self, name: str, size: int) -> float:
        """Average log-probability of the name's n-grams under the background."""
        counts = self.background_.trigram_counts
        total = sum(counts.values())
        if total == 0:
            return 0.0
        padded = START + name + END
        if len(padded) < size:
            return 0.0
        log_total = 0.0
        windows = 0
        for i in range(len(padded) - size + 1):
            gram = padded[i : i + size]
            if size == 2:
                # Sum the two trigrams that start with this bigram.
                prob = sum(
                    counts.get(gram + ch, 0)
                    for ch in "abcdefghijklmnopqrstuvwxyz^$"
                ) / total
            else:
                prob = counts.get(gram, 0) / total
            log_total += math.log(prob) if prob > 0 else math.log(1e-9)
            windows += 1
        return log_total / max(1, windows)


def _max_run(name: str, predicate) -> int:
    best = current = 0
    for ch in name:
        if predicate(ch):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


@dataclass
class ModelInfo:
    """Metadata about the currently active model."""

    trained_at: str = ""
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    cv_accuracy: Optional[float] = None
    cv_note: str = "not evaluated"
    feature_count: int = 0
    params: Dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        acc = (
            f"{self.cv_accuracy:.1%}" if self.cv_accuracy is not None else "n/a"
        )
        return (
            f"{self.n_samples} samples "
            f"({self.n_positive}+/{self.n_negative}-) | "
            f"CV accuracy {acc} | {self.feature_count} features"
        )


class NameClassifier:
    """A random forest that grades names as pass (1) / fail (0).

    The classifier always holds exactly one active model.  :meth:`fit` replaces
    it with a full refit over the full labelled dataset; :meth:`score` grades a
    single name; :meth:`grade` grades many.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        random_state: int = RANDOM_STATE,
        n_jobs: int = 1,
        cv_splits: int = DEFAULT_CV_SPLITS,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_splits = cv_splits
        self.pipeline: Optional[Pipeline] = None
        self.info = ModelInfo()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("features", NameFeatures()),
                (
                    "forest",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        bootstrap=True,
                    ),
                ),
            ]
        )

    def fit(self, dataset: Dataset, evaluate_cv: bool = True) -> ModelInfo:
        """Full refit on the entire dataset.

        This replaces the active model.  It does not compare against the
        previous model and does not keep the previous model - by design, so
        that the result depends only on the data currently in the CSV.

        Args:
            dataset: the labelled examples to train on.
            evaluate_cv: also compute the informational cross-validated
                accuracy (deterministic; never used to select a model).

        Returns:
            The :class:`ModelInfo` describing the newly trained model.
        """
        names = dataset.names
        labels = dataset.labels
        if len(names) == 0:
            raise ValueError("Cannot train on an empty dataset")
        if len(set(labels)) < 2:
            # A single-class forest still works, but every prediction is that
            # class, so say so plainly rather than pretending it is useful.
            self.info = ModelInfo(
                trained_at=_timestamp(),
                n_samples=len(names),
                n_positive=labels.count(1),
                n_negative=labels.count(0),
                cv_accuracy=None,
                cv_note="only one class present - scores are constant",
                feature_count=0,
                params=self._params(),
            )
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(names, labels)
            self.info.feature_count = self._feature_count()
            return self.info

        if evaluate_cv:
            cv_score, cv_note = self._cross_validate(names, labels)
        else:
            cv_score, cv_note = None, "not evaluated"

        pipeline = self._build_pipeline()
        pipeline.fit(names, labels)
        self.pipeline = pipeline

        pos, neg = dataset.class_balance()
        self.info = ModelInfo(
            trained_at=_timestamp(),
            n_samples=len(names),
            n_positive=pos,
            n_negative=neg,
            cv_accuracy=cv_score,
            cv_note=cv_note,
            feature_count=self._feature_count(),
            params=self._params(),
        )
        return self.info

    def _cross_validate(
        self, names: Sequence[str], labels: Sequence[int]
    ) -> Tuple[Optional[float], str]:
        """Informational CV accuracy with a fixed, reproducible splitter."""
        n_samples = len(names)
        min_class = min(labels.count(1), labels.count(0))
        if n_samples < 4 or min_class < 2:
            return None, "not enough data for cross-validation"
        splits = min(self.cv_splits, min_class)
        if splits < 2:
            return None, "not enough data for cross-validation"
        try:
            cv = StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                self._build_pipeline(), list(names), list(labels), cv=cv, n_jobs=1
            )
            return float(scores.mean()), f"{splits}-fold CV (informational only)"
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"CV failed: {exc}"

    def _feature_count(self) -> int:
        if self.pipeline is None:
            return 0
        try:
            features = self.pipeline.named_steps["features"]
            return len(features.get_feature_names_out())
        except Exception:  # pragma: no cover - defensive
            return 0

    def _params(self) -> Dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def score(self, name: str) -> float:
        """Return ``P(pass)`` for a single name.

        With no trained model this returns 0.0 (nothing has been learned yet,
        so nothing passes).
        """
        if self.pipeline is None:
            return 0.0
        proba = self.pipeline.predict_proba([name])
        classes = list(self.pipeline.named_steps["forest"].classes_)
        if 1 not in classes:
            return 0.0
        return float(proba[0][classes.index(1)])

    def grade(
        self, names: Sequence[str], threshold: float = 0.5
    ) -> List[Tuple[str, float, bool]]:
        """Grade many names at once.

        Args:
            names: the names to score.
            threshold: minimum score for a name to count as a pass.

        Returns:
            List of ``(name, score, passes)`` where ``passes`` is
            ``score >= threshold``.
        """
        out: List[Tuple[str, float, bool]] = []
        if not names:
            return out
        if self.pipeline is None:
            return [(n, 0.0, False) for n in names]
        proba = self.pipeline.predict_proba(list(names))
        classes = list(self.pipeline.named_steps["forest"].classes_)
        idx = classes.index(1) if 1 in classes else None
        for name, row in zip(names, proba):
            score = float(row[idx]) if idx is not None else 0.0
            out.append((name, score, score >= threshold))
        return out

    def feature_importances(self, top: int = 12) -> List[Tuple[str, float]]:
        """Return the most influential features, for the UI's explain panel."""
        if self.pipeline is None:
            return []
        try:
            forest = self.pipeline.named_steps["forest"]
            features = self.pipeline.named_steps["features"]
            names = list(features.get_feature_names_out())
            values = forest.feature_importances_
            pairs = list(zip(names, values))
            pairs.sort(key=lambda p: p[1], reverse=True)
            return pairs[:top]
        except Exception:  # pragma: no cover - defensive
            return []


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_training_set(
    user_examples: Sequence, seed_examples: Sequence[Tuple[str, int]] = ()
) -> Dataset:
    """Combine the user's labels with the seed baseline into one training set.

    Both interfaces call this so they train on exactly the same rows.  The seed
    rows are added first and are never written to the user's CSV; because
    training is a plain full refit they carry the same weight as any other
    row.
    """
    combined = Dataset()
    for name, label in seed_examples:
        combined.add(name, label)
    combined.extend(user_examples)
    return combined


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_model(classifier: NameClassifier, path: str) -> str:
    """Persist a trained classifier to disk with joblib."""
    import joblib

    if classifier.pipeline is None:
        raise ValueError("No trained model to save")
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "pipeline": classifier.pipeline,
        "info": classifier.info,
        "params": classifier._params(),
    }
    joblib.dump(payload, path)
    return os.path.abspath(path)


def load_model(path: str) -> NameClassifier:
    """Load a classifier previously saved with :func:`save_model`."""
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    classifier = NameClassifier(**payload.get("params", {}))
    classifier.pipeline = payload["pipeline"]
    classifier.info = payload.get("info", ModelInfo())
    return classifier
