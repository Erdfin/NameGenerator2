"""Ultra-simplified grading interface.

One name at a time, four buttons, nothing else::

    +---------------------------------------------+
    |                                             |
    |                 elenarion                   |
    |          model score 0.87  |  42 left        |
    |                                             |
    |   [ Accept (A) ]  [ Reject (R) ]             |
    |   [  Skip (S)   ]  [  Retrain (T) ]         |
    |                                             |
    |  Recorded 'elena' as pass                    |
    +---------------------------------------------+

Only names the model *passed* are shown, so the accept / reject decision is
always a meaningful override of the model's opinion.  Skip records nothing and
simply moves on.

Key bindings (work anywhere in the window):

===========  ==========  ==========================================
Key          Button      Action
===========  ==========  ==========================================
``A``        Accept      record the name as a pass
``R``        Reject      record the name as a fail
``S``        Skip        record nothing, show the next name
``T``        Retrain     full refit of the model
``Return``   Accept      same as ``A``
``Space``    Accept      same as ``A``
``Q``        -           quit
``Escape``   -           quit
===========  ==========  ==========================================
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox
from typing import List, Optional, Tuple

from dataset import Dataset, DatasetError, append_example, load_dataset
from generator import NameGenerator
from model import NameClassifier, build_training_set
from seed_data import seed_dataset

DEFAULT_DATA_FILE = "data.csv"

#: How many candidates to generate and score at a time.
BATCH_SIZE = 60

#: The pass threshold used to decide which names count as "passed".
DEFAULT_THRESHOLD = 0.5

#: Colour used for a name the model is confident about.
GOOD_COLOUR = "#1b7f3b"
#: Colour used when the model is not sure.
WEAK_COLOUR = "#8a6d00"
#: Colour used when a name is shown despite not clearing the threshold.
FALLBACK_COLOUR = "#9a3412"


class SimpleGraderApp:
    """The minimal grading window."""

    def __init__(
        self,
        root: tk.Tk,
        data_file: str = DEFAULT_DATA_FILE,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.root = root
        self.root.title("Name Grader")
        self.root.geometry("620x420")
        self.root.minsize(480, 360)
        self.root.configure(bg="#fbfbfd")

        self.data_file = os.path.abspath(data_file)
        self.threshold = threshold

        self.dataset = Dataset()
        self.seed_rows = list(seed_dataset())
        self.classifier = NameClassifier()
        self.generator: Optional[NameGenerator] = None

        self.queue: List[Tuple[str, float]] = []
        self.current: Optional[str] = None
        self.current_score: float = 0.0
        self.shown: set = set()
        self.skipped = 0
        self.accepted = 0
        self.rejected = 0

        self.name_var = tk.StringVar(value="")
        self.meta_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Starting up...")
        self.counts_var = tk.StringVar(value="")

        self._build_widgets()
        self._bind_keys()
        self._load_dataset_file(self.data_file, quiet=True)
        self._retrain(quiet=True)
        self._refill_queue()
        self._show_next()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        bg = self.root["bg"]

        self.name_label = tk.Label(
            self.root,
            textvariable=self.name_var,
            font=("Segoe UI", 44, "bold"),
            bg=bg,
            fg="#111827",
        )
        self.name_label.pack(pady=(44, 4))

        tk.Label(
            self.root,
            textvariable=self.meta_var,
            font=("Segoe UI", 11),
            bg=bg,
            fg="#6b7280",
        ).pack()

        tk.Label(
            self.root,
            textvariable=self.counts_var,
            font=("Segoe UI", 10),
            bg=bg,
            fg="#9ca3af",
        ).pack(pady=(2, 0))

        # -- buttons ----------------------------------------------------
        button_frame = tk.Frame(self.root, bg=bg)
        button_frame.pack(pady=34)

        self.accept_btn = self._make_button(
            button_frame, "Accept  (A)", self._on_accept, "#16a34a", "white", 0, 0
        )
        self.reject_btn = self._make_button(
            button_frame, "Reject  (R)", self._on_reject, "#dc2626", "white", 0, 1
        )
        self.skip_btn = self._make_button(
            button_frame, "Skip  (S)", self._on_skip, "#6b7280", "white", 1, 0
        )
        self.retrain_btn = self._make_button(
            button_frame, "Retrain  (T)", self._on_retrain, "#2563eb", "white", 1, 1
        )

        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg=bg,
            fg="#4b5563",
            wraplength=540,
        ).pack(side="bottom", pady=(0, 18))

        tk.Label(
            self.root,
            text="A accept  ·  R reject  ·  S skip  ·  T retrain  ·  Q quit",
            font=("Segoe UI", 9),
            bg=bg,
            fg="#9ca3af",
        ).pack(side="bottom", pady=(0, 8))

    def _make_button(
        self, parent, text, command, bg, fg, row, col
    ) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 12, "bold"),
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=26,
            pady=12,
            cursor="hand2",
            takefocus=0,
            highlightthickness=0,
        )
        btn.grid(row=row, column=col, padx=8, pady=6)
        return btn

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-a>", lambda _e: self._on_accept())
        self.root.bind("<KeyPress-A>", lambda _e: self._on_accept())
        self.root.bind("<KeyPress-r>", lambda _e: self._on_reject())
        self.root.bind("<KeyPress-R>", lambda _e: self._on_reject())
        self.root.bind("<KeyPress-s>", lambda _e: self._on_skip())
        self.root.bind("<KeyPress-S>", lambda _e: self._on_skip())
        self.root.bind("<KeyPress-t>", lambda _e: self._on_retrain())
        self.root.bind("<KeyPress-T>", lambda _e: self._on_retrain())
        self.root.bind("<Return>", lambda _e: self._on_accept())
        self.root.bind("<Key-space>", lambda _e: self._on_accept())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("<KeyPress-q>", lambda _e: self.root.destroy())
        self.root.bind("<KeyPress-Q>", lambda _e: self.root.destroy())
        self.root.focus_set()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def _load_dataset_file(self, path: str, quiet: bool = False) -> None:
        try:
            if os.path.exists(path):
                self.dataset = load_dataset(path)
                msg = f"Loaded {len(self.dataset)} of your labels"
            else:
                self.dataset = Dataset()
                msg = "No file yet - your labels will be written to it"
            self.data_file = os.path.abspath(path)
            self._set_status(msg)
        except DatasetError as exc:
            self._set_status(f"Could not load dataset: {exc}")
            if not quiet:
                messagebox.showerror("Dataset error", str(exc))

    def _training_set(self) -> Dataset:
        return build_training_set(self.dataset.examples, self.seed_rows)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _retrain(self, quiet: bool = False) -> None:
        training = self._training_set()
        if len(training) == 0:
            self._set_status("Nothing to train on")
            return
        try:
            info = self.classifier.fit(training)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status(f"Training failed: {exc}")
            return
        if not quiet:
            acc = (
                f"{info.cv_accuracy:.1%}"
                if info.cv_accuracy is not None
                else "n/a"
            )
            self._set_status(
                f"Retrained on {info.n_samples} rows (CV accuracy {acc}) - "
                "regenerating candidates"
            )
        # The model changed, so the old candidates are stale.
        self._refill_queue()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _rebuild_generator(self) -> None:
        positives = [e.name for e in self.dataset.positives()]
        if len(positives) < 2:
            positives += [n for n, label in self.seed_rows if label == 1]
        self.generator = NameGenerator(positives=positives)

    def _refill_queue(self) -> None:
        """Generate a fresh batch and keep the names the model passed."""
        if self.generator is None:
            self._rebuild_generator()
        self.queue = []
        self.shown = set()

        names = self.generator.generate(
            mode="random", count=BATCH_SIZE, min_length=4, max_length=10
        )
        if not names:
            self._set_status("Could not generate any names")
            return

        graded = self.classifier.grade(names, threshold=self.threshold)
        scored = [(name, score) for name, score, _ in graded]
        # Best first, so the strongest candidates are seen soonest.
        scored.sort(key=lambda pair: pair[1], reverse=True)

        passing = [pair for pair in scored if pair[1] >= self.threshold]
        self.queue = passing if passing else scored[:10]
        self._all_fell_below = not passing

    def _show_next(self) -> None:
        if not self.queue:
            self._refill_queue()
        if not self.queue:
            self.name_var.set("---")
            self.meta_var.set("no candidates available")
            return

        name, score = self.queue.pop(0)
        self.current = name
        self.current_score = score
        self.shown.add(name)

        self.name_var.set(name)
        if score >= self.threshold:
            self.name_label.config(fg=GOOD_COLOUR if score >= 0.8 else WEAK_COLOUR)
            self.meta_var.set(f"model score {score:.2f}")
        else:
            self.name_label.config(fg=FALLBACK_COLOUR)
            self.meta_var.set(
                f"model score {score:.2f} - below threshold, shown anyway"
            )
        self._update_counts()

    def _update_counts(self) -> None:
        self.counts_var.set(
            f"{self.accepted} accepted  ·  {self.rejected} rejected  ·  "
            f"{self.skipped} skipped  ·  {len(self.queue)} queued"
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_accept(self) -> None:
        if not self.current:
            return
        name = self.current
        if self._record(name, 1):
            self.accepted += 1
            self._set_status(f"Recorded '{name}' as pass")
        self._show_next()

    def _on_reject(self) -> None:
        if not self.current:
            return
        name = self.current
        if self._record(name, 0):
            self.rejected += 1
            self._set_status(f"Recorded '{name}' as fail")
        self._show_next()

    def _on_skip(self) -> None:
        if not self.current:
            return
        name = self.current
        self.skipped += 1
        self._set_status(f"Skipped '{name}' (nothing recorded)")
        self._show_next()

    def _on_retrain(self) -> None:
        self._retrain()
        self._show_next()

    def _record(self, name: str, label: int) -> bool:
        """Append a grade to the CSV and keep memory in sync."""
        try:
            path = append_example(self.data_file, name, label)
        except (DatasetError, OSError) as exc:
            messagebox.showerror("Could not save", str(exc))
            return False
        self.dataset.add(name, label)
        self.data_file = path
        self._rebuild_generator()
        return True

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def launch(
    data_file: str = DEFAULT_DATA_FILE, threshold: float = DEFAULT_THRESHOLD
) -> None:
    """Start the simplified grading interface."""
    root = tk.Tk()
    SimpleGraderApp(root, data_file=data_file, threshold=threshold)
    root.mainloop()
