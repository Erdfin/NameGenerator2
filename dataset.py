"""CSV dataset handling for the name generator.

Canonical CSV format (two columns)::

    name,label
    alice,1
    xkcdzx,0

The loader is deliberately tolerant so hand-made CSVs work: the name column may
be called ``name``/``text``/``word``/``label_name`` and the label column may be
``label``/``target``/``class``/``y``.  Labels may be written as ``1``/``0``,
``pass``/``fail``, ``yes``/``no``, ``true``/``false``, ``pos``/``neg``,
``good``/``bad`` or ``accepted``/``rejected``.

A seed corpus lives in :mod:`seed_data` and is only used as a fallback when no
user dataset exists yet.  It is never merged into the user's own ``data.csv``
so that the user's file only ever contains labels they gave themselves.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

# Label vocabulary -> boolean "is a positive (pass)" mapping.
POSITIVE_TOKENS = {
    "1", "true", "yes", "y", "t", "pass", "passed", "pos", "positive",
    "good", "ok", "accept", "accepted", "nice", "like", "liked",
}
NEGATIVE_TOKENS = {
    "0", "false", "no", "n", "f", "fail", "failed", "neg", "negative",
    "bad", "reject", "rejected", "nope", "dislike", "disliked",
}

NAME_HEADERS = ("name", "text", "word", "label_name", "example", "string", "value")
LABEL_HEADERS = ("label", "target", "class", "y", "is_good", "result", "verdict")


class DatasetError(Exception):
    """Raised when a dataset file cannot be read or is structurally wrong."""


@dataclass
class Example:
    """A single labelled name."""

    name: str
    label: int  # 1 = pass, 0 = fail

    def as_row(self) -> Tuple[str, str]:
        return self.name, str(self.label)


@dataclass
class Dataset:
    """An ordered collection of labelled examples."""

    examples: List[Example] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def names(self) -> List[str]:
        return [e.name for e in self.examples]

    @property
    def labels(self) -> List[int]:
        return [e.label for e in self.examples]

    def positives(self) -> List[Example]:
        return [e for e in self.examples if e.label == 1]

    def negatives(self) -> List[Example]:
        return [e for e in self.examples if e.label == 0]

    def add(self, name: str, label: int) -> Example:
        ex = Example(name=name.strip(), label=int(label))
        self.examples.append(ex)
        return ex

    def extend(self, examples: Iterable[Example]) -> None:
        for ex in examples:
            self.examples.append(ex)

    def as_rows(self) -> List[Tuple[str, str]]:
        return [e.as_row() for e in self.examples]

    def class_balance(self) -> Tuple[int, int]:
        """Return ``(n_positive, n_negative)``."""
        pos = sum(1 for e in self.examples if e.label == 1)
        return pos, len(self.examples) - pos

    def save(self, path: str) -> str:
        """Write the dataset to ``path`` in canonical format."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "label"])
            for ex in self.examples:
                writer.writerow(ex.as_row())
        return os.path.abspath(path)


def normalise_label(raw: object) -> int:
    """Convert any supported label spelling into ``1`` (pass) or ``0`` (fail).

    Raises:
        DatasetError: if the value cannot be interpreted.
    """
    if raw is None:
        raise DatasetError("Missing label value")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) in (0, 1):
            return int(raw)
        raise DatasetError(f"Label must be 0 or 1, got {raw!r}")

    text = str(raw).strip().lower()
    if text in POSITIVE_TOKENS:
        return 1
    if text in NEGATIVE_TOKENS:
        return 0
    if text in {"1.0", "0.0"}:
        return int(float(text))
    raise DatasetError(
        f"Unrecognised label {raw!r}. Use 1/0, pass/fail, yes/no, true/false, "
        "pos/neg, good/bad or accepted/rejected."
    )


def _clean_name(raw: object) -> str:
    return str(raw).strip().lower()


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_dataset(path: str) -> Dataset:
    """Load a labelled dataset from a CSV file.

    The file may have a header row or not.  If it has no header, the first two
    columns are interpreted as (name, label).

    Raises:
        DatasetError: if the file is missing, empty or unparseable.
    """
    if not os.path.exists(path):
        raise DatasetError(f"Dataset file not found: {path}")

    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        raise DatasetError(f"Dataset file is empty: {path}")

    header = [cell.strip() for cell in rows[0]]
    name_col = _pick_column(header, NAME_HEADERS)
    label_col = _pick_column(header, LABEL_HEADERS)
    has_header = name_col is not None or label_col is not None

    if has_header:
        if name_col is None:
            name_col = header[0]
        if label_col is None:
            # Fall back to the first non-name column.
            for candidate in header:
                if candidate != name_col:
                    label_col = candidate
                    break
        name_idx = header.index(name_col)
        label_idx = header.index(label_col) if label_col in header else 1
        data_rows = rows[1:]
    else:
        name_idx, label_idx = 0, 1
        data_rows = rows

    dataset = Dataset()
    for lineno, row in enumerate(data_rows, start=2 if has_header else 1):
        if name_idx >= len(row):
            continue
        name = _clean_name(row[name_idx])
        if not name:
            continue
        if label_idx >= len(row):
            raise DatasetError(
                f"{path}: row {lineno} has a name but no label column value"
            )
        label = normalise_label(row[label_idx])
        dataset.add(name, label)

    return dataset


def append_example(path: str, name: str, label: int) -> str:
    """Append a single labelled example to a CSV file, creating it if needed.

    If the file exists and already has a header, the row is appended in the
    file's own column order.  Otherwise a canonical header is written first.
    """
    name = name.strip()
    if not name:
        raise DatasetError("Cannot append an empty name")

    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if new_file:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "label"])
            writer.writerow([name, int(label)])
        return os.path.abspath(path)

    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.reader(fh) if any(c.strip() for c in row)]
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "label"])
            writer.writerow([name, int(label)])
        return os.path.abspath(path)

    header = [cell.strip() for cell in rows[0]]
    name_col = _pick_column(header, NAME_HEADERS) or header[0]
    label_col = _pick_column(header, LABEL_HEADERS)
    if label_col is None:
        for candidate in header:
            if candidate != name_col:
                label_col = candidate
                break
    if label_col is None:
        label_col = "label"
    name_idx = header.index(name_col)
    label_idx = header.index(label_col) if label_col in header else 1

    row = [""] * max(len(header), name_idx + 1, label_idx + 1)
    row[name_idx] = name
    row[label_idx] = str(int(label))

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(row)
    return os.path.abspath(path)
