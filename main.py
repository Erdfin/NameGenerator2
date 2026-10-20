"""Entry point for the name generator.

Usage::

    python main.py                    # launch the tkinter GUI
    python main.py --data my.csv      # use a specific dataset file
    python main.py --selftest         # headless end-to-end check
    python main.py --corpus words.txt # use a custom corpus for generation
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import List

from dataset import Dataset, DatasetError, append_example, load_dataset
from generator import NameGenerator, load_corpus_from_file
from model import NameClassifier
from seed_data import seed_dataset

DEFAULT_DATA_FILE = "data.csv"

_counter = itertools.count()


def _work_dir() -> str:
    """A writable scratch directory for the self-test.

    The system temp directory is not always writable (sandboxes, locked-down
    Windows profiles), so the scratch space lives next to the project.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".selftest_tmp")
    os.makedirs(path, exist_ok=True)
    return path


def _remove_tree(path: str) -> None:
    """Delete a directory tree without the ``chmod`` that ``shutil.rmtree`` does."""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass


class _Scratch:
    """A temporary directory that cleans up without calling ``chmod``.

    Two sandbox quirks are worked around here:

    * ``tempfile.mkdtemp`` creates directories with mode ``0o700``, and some
      sandboxes then refuse writes inside them - so a plain ``os.makedirs``
      with default permissions is used instead.
    * ``tempfile.TemporaryDirectory`` calls ``shutil.rmtree``, which resets
      permissions on every file before deleting it, and that ``chmod`` is
      blocked in some environments - so removal is done manually.
    """

    def __init__(self, parent: str, prefix: str = "scratch_") -> None:
        unique = f"{prefix}{os.getpid()}_{next(_counter)}"
        self.path = os.path.join(parent, unique)
        os.makedirs(self.path, exist_ok=True)

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, *_exc) -> None:
        _remove_tree(self.path)


def _selftest_simple(scratch: str, failures: List[str]) -> None:
    """Headless checks for the simplified grading interface."""
    try:
        import tkinter as tk

        from ui_simple import SimpleGraderApp
    except Exception as exc:
        failures.append(f"simple interface could not be imported: {exc}")
        return

    path = os.path.join(scratch, "simple.csv")
    root = tk.Tk()
    root.withdraw()
    try:
        app = SimpleGraderApp(root, data_file=path)
        root.update()
    except Exception as exc:
        failures.append(f"simple interface failed to start: {exc}")
        root.destroy()
        return

    def key(keysym: str) -> None:
        root.event_generate(f"<KeyPress-{keysym}>", when="now")
        root.update()

    try:
        # A name must be on screen, and it must be one the model passed.
        if not app.current:
            failures.append("simple interface showed no name on startup")
        else:
            if app.current_score < app.threshold and not getattr(
                app, "_all_fell_below", False
            ):
                failures.append(
                    f"simple interface showed a below-threshold name "
                    f"({app.current} @ {app.current_score:.2f})"
                )
        print(f"  [ok] simple interface started, showing '{app.current}'")

        # Accept via the 'a' key must write a pass row.
        before = len(app.dataset)
        key("a")
        if len(app.dataset) != before + 1:
            failures.append("key 'a' did not record an accept")
        elif app.dataset.labels[-1] != 1:
            failures.append("key 'a' recorded a label other than pass")
        print("  [ok] key 'a' records a pass")

        # Reject via the 'r' key must write a fail row.
        before = len(app.dataset)
        key("r")
        if len(app.dataset) != before + 1:
            failures.append("key 'r' did not record a reject")
        elif app.dataset.labels[-1] != 0:
            failures.append("key 'r' recorded a label other than fail")
        print("  [ok] key 'r' records a fail")

        # Skip must record nothing.
        before = len(app.dataset)
        key("s")
        if len(app.dataset) != before:
            failures.append("key 's' recorded a row (skip must not record)")
        print("  [ok] key 's' records nothing")

        # Retrain via the 't' key must produce a trained model.
        key("t")
        if not app.classifier.is_trained():
            failures.append("key 't' left the model untrained")
        print("  [ok] key 't' retrains")

        # Space and Return are accept aliases.
        before = len(app.dataset)
        key("space")
        if len(app.dataset) != before + 1:
            failures.append("key 'space' did not record an accept")
        before = len(app.dataset)
        key("Return")
        if len(app.dataset) != before + 1:
            failures.append("key 'Return' did not record an accept")
        print("  [ok] space and Return are accept aliases")

        # The CSV on disk must match what the app recorded.
        on_disk = load_dataset(path)
        if len(on_disk) != len(app.dataset):
            failures.append(
                f"CSV on disk has {len(on_disk)} rows but the app has "
                f"{len(app.dataset)}"
            )
        print(f"  [ok] CSV on disk matches memory ({len(on_disk)} rows)")
    finally:
        root.destroy()


def _selftest() -> int:
    """Run an end-to-end check without a GUI."""
    print("== Name generator self-test ==")
    failures = []
    scratch = _work_dir()

    # 1. Dataset round-trip.
    with _Scratch(scratch) as tmp:
        path = os.path.join(tmp, "test.csv")
        append_example(path, "alice", 1)
        append_example(path, "xkcdzx", 0)
        append_example(path, "bob", 1)
        ds = load_dataset(path)
        if len(ds) != 3:
            failures.append(f"dataset round-trip: expected 3 rows, got {len(ds)}")
        if ds.names != ["alice", "xkcdzx", "bob"]:
            failures.append(f"dataset names mismatch: {ds.names}")
        print(f"  [ok] dataset round-trip ({len(ds)} rows)")

        # 2. Tolerant label parsing.
        from dataset import normalise_label

        cases = {"1": 1, "pass": 1, "YES": 1, "good": 1, "accepted": 1,
                 "0": 0, "fail": 0, "no": 0, "bad": 0, "rejected": 0, "pos": 1,
                 "neg": 0}
        for raw, expected in cases.items():
            got = normalise_label(raw)
            if got != expected:
                failures.append(f"label {raw!r}: expected {expected}, got {got}")
        print(f"  [ok] label normalisation ({len(cases)} spellings)")

    # 3. Train on the seed corpus.
    dataset = Dataset()
    for name, label in seed_dataset():
        dataset.add(name, label)
    classifier = NameClassifier()
    info = classifier.fit(dataset)
    print(
        f"  [ok] trained on {info.n_samples} seed rows, "
        f"CV accuracy {info.cv_accuracy:.1%} ({info.cv_note})"
    )
    if info.cv_accuracy is None:
        failures.append("CV accuracy was not computed")

    # 4. Determinism: same data -> identical model -> identical scores.
    classifier2 = NameClassifier()
    classifier2.fit(dataset)
    names = ["alice", "xkcdzx", "maria", "qwerty", "bruno"]
    scores_a = [classifier.score(n) for n in names]
    scores_b = [classifier2.score(n) for n in names]
    if scores_a != scores_b:
        failures.append(f"retrain is not deterministic: {scores_a} != {scores_b}")
    print(f"  [ok] deterministic retrain (scores identical: {scores_a})")

    # 5. The model should separate obvious passes from obvious fails.
    good = [n for n in dataset.positives()[:20]]
    bad = [n for n in dataset.negatives()[:20]]
    good_scores = [classifier.score(n) for n in good]
    bad_scores = [classifier.score(n) for n in bad]
    if sum(good_scores) / len(good_scores) <= sum(bad_scores) / len(bad_scores):
        failures.append(
            "model does not separate known good from known bad names"
        )
    print(
        f"  [ok] separates classes: mean good {sum(good_scores)/len(good_scores):.3f} "
        f"vs mean bad {sum(bad_scores)/len(bad_scores):.3f}"
    )

    # 6. Generator produces usable names in every mode.
    generator = NameGenerator(positives=[e.name for e in dataset.positives()])
    for mode in ("random", "markov"):
        produced = generator.generate(mode=mode, count=25, min_length=4, max_length=10)
        if len(produced) < 5:
            failures.append(f"mode {mode}: only produced {len(produced)} names")
        if any(not n.isalpha() for n in produced):
            failures.append(f"mode {mode}: produced non-alphabetic names")
        print(f"  [ok] generator mode '{mode}' produced {len(produced)} names")

    # 7. 'accepted' mode keeps only names the model passes.
    accepted = generator.generate_until(
        scorer=classifier.score, mode="random", wanted=3
    )
    if not accepted:
        failures.append("accepted mode found nothing")
    else:
        bad_accept = [n for n, s in accepted if s < 0.5]
        if bad_accept:
            failures.append(f"accepted mode returned failing names: {bad_accept}")
        print(f"  [ok] accepted mode returned {len(accepted)} passing names")

    # 8. Grading appends to the CSV and the new row is used on retrain.
    with _Scratch(scratch) as tmp:
        path = os.path.join(tmp, "live.csv")
        append_example(path, "alice", 1)
        append_example(path, "xkcdzx", 0)
        append_example(path, "maria", 1)
        append_example(path, "qwerty", 0)
        before = load_dataset(path)
        append_example(path, "bruno", 1)
        after = load_dataset(path)
        if len(after) != len(before) + 1:
            failures.append("appending a grade did not grow the dataset")
        if after.names[-1] != "bruno" or after.labels[-1] != 1:
            failures.append("appended row has the wrong contents")
        print("  [ok] grading appends to the CSV")

    # 9. Single-class dataset must not crash.
    single = Dataset()
    single.add("solo", 1)
    try:
        NameClassifier().fit(single)
        print("  [ok] single-class dataset handled")
    except Exception as exc:
        failures.append(f"single-class dataset crashed: {exc}")

    # 10. The simplified interface (headless tkinter).
    _selftest_simple(scratch, failures)

    print()
    _remove_tree(scratch)
    if failures:
        print(f"FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All self-tests passed.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Random-forest graded name generator")
    parser.add_argument(
        "--data", default=DEFAULT_DATA_FILE, help="path to the labelled CSV"
    )
    parser.add_argument(
        "--corpus", default=None, help="optional text corpus for trigram generation"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="launch the ultra-simplified one-name-at-a-time grader",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="pass threshold for --simple (default: 0.5)",
    )
    parser.add_argument(
        "--selftest", action="store_true", help="run headless checks and exit"
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    corpus = None
    if args.corpus:
        try:
            corpus = load_corpus_from_file(args.corpus)
        except OSError as exc:
            print(f"Could not read corpus: {exc}", file=sys.stderr)
            return 2
        print(f"Loaded {len(corpus)} words from {args.corpus}")

    try:
        if args.simple:
            from ui_simple import launch as launch_simple

            launch_simple(data_file=args.data, threshold=args.threshold)
        else:
            from ui import launch

            launch(data_file=args.data, corpus=corpus)
    except ImportError as exc:
        print(f"Could not start the GUI: {exc}", file=sys.stderr)
        print("Install the dependencies with: pip install -r requirements.txt",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
