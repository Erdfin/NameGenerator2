"""Tkinter GUI for the name generator.

Layout::

    +-------------------------------------------------------------+
    | Dataset file: [ path ....... ] [Load] [Use seed] [Save]      |
    | Mode [random v] Length [4]-[10] Threshold [0.50]             |
    | [Generate 1] [Generate 10] [Generate until 5 pass]           |
    +---------------------------+---------------------------------+
    | Candidates               | Status / model panel              |
    |  Name | Score | Verdict   |  samples, balance, CV accuracy    |
    |  ...                      |  [x] Pass  [x] Fail  (grade)      |
    |                           |  [Retrain model]                   |
    +---------------------------+---------------------------------+
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

from dataset import (
    Dataset,
    DatasetError,
    append_example,
    load_dataset,
)
from generator import NameGenerator
from model import NameClassifier, build_training_set, save_model
from seed_data import seed_dataset

DEFAULT_DATA_FILE = "data.csv"
DEFAULT_MODEL_FILE = "model.joblib"
DEFAULT_THRESHOLD = 0.5

MODE_LABELS = {
    "random": "Random trigrams",
    "markov": "Markov (from passes)",
    "accepted": "Generate until accepted",
}
MODE_VALUES = {
    "Random trigrams": "random",
    "Markov (from passes)": "markov",
    "Generate until accepted": "accepted",
}


class NameGeneratorApp:
    """The main application window."""

    def __init__(self, root: tk.Tk, data_file: str = DEFAULT_DATA_FILE) -> None:
        self.root = root
        self.root.title("Name Generator - random forest grader")
        self.root.geometry("900x600")
        self.root.minsize(760, 480)

        self.data_file = os.path.abspath(data_file)
        self.dataset = Dataset()
        self.seed_rows: List[Tuple[str, int]] = list(seed_dataset())
        self.classifier = NameClassifier()
        self.generator: Optional[NameGenerator] = None
        self.candidates: List[Tuple[str, float, bool]] = []
        self.corpus_words: Optional[List[str]] = None
        self._loading = False

        # -- widget state ------------------------------------------------
        self.data_path_var = tk.StringVar(value=self.data_file)
        self.mode_var = tk.StringVar(value="Random trigrams")
        self.min_len_var = tk.IntVar(value=4)
        self.max_len_var = tk.IntVar(value=10)
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.status_var = tk.StringVar(value="Starting up...")
        self.model_var = tk.StringVar(value="No model trained yet")
        self.dataset_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")

        self._build_widgets()
        self._load_dataset_file(self.data_file, quiet=True)
        self._retrain(quiet=True)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        pad = {"padx": 6, "pady": 4}

        top = ttk.LabelFrame(self.root, text="Dataset")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="CSV:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(top, textvariable=self.data_path_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(top, text="Load...", command=self._on_load).grid(
            row=0, column=2, padx=4, pady=4
        )
        ttk.Button(top, text="Save as...", command=self._on_save_as).grid(
            row=0, column=3, padx=4, pady=4
        )
        ttk.Button(top, text="Retrain", command=self._retrain).grid(
            row=0, column=4, padx=4, pady=4
        )
        ttk.Label(top, textvariable=self.dataset_var).grid(
            row=1, column=0, columnspan=5, sticky="w", padx=4, pady=(0, 4)
        )

        controls = ttk.LabelFrame(self.root, text="Generator")
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(controls, text="Mode:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=list(MODE_LABELS.keys()),
            state="readonly",
            width=22,
        )
        mode_box.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_change())

        ttk.Label(controls, text="Length:").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Spinbox(
            controls, from_=2, to=20, width=4, textvariable=self.min_len_var
        ).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(controls, text="to").grid(row=0, column=4, padx=4)
        ttk.Spinbox(
            controls, from_=2, to=20, width=4, textvariable=self.max_len_var
        ).grid(row=0, column=5, sticky="w", padx=4)

        ttk.Label(controls, text="Pass threshold:").grid(
            row=0, column=6, sticky="e", padx=4
        )
        ttk.Spinbox(
            controls,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=5,
            textvariable=self.threshold_var,
            command=self._on_threshold_change,
        ).grid(row=0, column=7, sticky="w", padx=4)

        buttons = ttk.Frame(controls)
        buttons.grid(row=1, column=0, columnspan=8, sticky="w", padx=4, pady=(2, 6))
        ttk.Button(buttons, text="Generate 1", command=self._on_generate_one).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(buttons, text="Generate 10", command=self._on_generate_ten).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(
            buttons, text="Generate until 5 pass", command=self._on_generate_accepted
        ).pack(side="left")

        # -- candidates + side panel ------------------------------------
        body = ttk.Frame(self.root)
        body.grid(row=2, column=0, columnspan=2, sticky="nsew", **pad)
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)

        left = ttk.LabelFrame(body, text="Candidates")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        columns = ("name", "score", "verdict")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=18)
        self.tree.heading("name", text="Name")
        self.tree.heading("score", text="Score")
        self.tree.heading("verdict", text="Verdict")
        self.tree.column("name", width=220, anchor="w")
        self.tree.column("score", width=80, anchor="e")
        self.tree.column("verdict", width=90, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<Double-1>", lambda _e: self._on_grade(1))

        right = ttk.LabelFrame(body, text="Grade & model")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        ttk.Label(right, textvariable=self.model_var, wraplength=320).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 2)
        )
        ttk.Label(right, textvariable=self.detail_var, wraplength=320).grid(
            row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 8)
        )

        grade = ttk.Frame(right)
        grade.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6)
        ttk.Label(grade, text="Selected name:").grid(row=0, column=0, sticky="w")
        self.selected_var = tk.StringVar(value="(none selected)")
        ttk.Label(grade, textvariable=self.selected_var).grid(
            row=0, column=1, columnspan=2, sticky="w", padx=4
        )
        ttk.Button(grade, text="Pass", command=lambda: self._on_grade(1)).grid(
            row=1, column=0, sticky="ew", pady=(8, 0), padx=(0, 4)
        )
        ttk.Button(grade, text="Fail", command=lambda: self._on_grade(0)).grid(
            row=1, column=1, sticky="ew", pady=(8, 0), padx=4
        )
        ttk.Button(grade, text="Save model", command=self._on_save_model).grid(
            row=1, column=2, sticky="ew", pady=(8, 0), padx=(4, 0)
        )

        ttk.Separator(right).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=10
        )

        ttk.Label(right, text="Top features:").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=6
        )
        self.importance_text = tk.Text(right, height=8, width=40, wrap="none")
        self.importance_text.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=6)
        right.rowconfigure(5, weight=1)

        ttk.Separator(right).grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=6, pady=10
        )
        ttk.Label(right, textvariable=self.status_var, wraplength=320).grid(
            row=7, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6)
        )

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def _load_dataset_file(self, path: str, quiet: bool = False) -> None:
        self._loading = True
        try:
            if os.path.exists(path):
                self.dataset = load_dataset(path)
                self.data_file = os.path.abspath(path)
                self.data_path_var.set(self.data_file)
                msg = f"Loaded {len(self.dataset)} rows from {os.path.basename(path)}"
            else:
                # Cold start: the user's CSV stays empty and purely their own
                # labels.  The built-in seed corpus is used as a training
                # baseline instead (see _training_set).
                self.dataset = Dataset()
                self.data_file = os.path.abspath(path)
                self.data_path_var.set(self.data_file)
                msg = (
                    f"No {os.path.basename(path)} yet - starting with the built-in "
                    f"seed corpus as a baseline. Your grades will be written to "
                    "this file."
                )
            self._update_dataset_label()
            self._rebuild_generator()
            self._set_status(msg)
        except DatasetError as exc:
            self._set_status(f"Could not load dataset: {exc}")
            if not quiet:
                messagebox.showerror("Dataset error", str(exc))
        finally:
            self._loading = False

    def _update_dataset_label(self) -> None:
        pos, neg = self.dataset.class_balance()
        self.dataset_var.set(
            f"{len(self.dataset)} of your examples  |  {pos} pass / {neg} fail  |  "
            f"trains on {len(self.dataset) + len(self.seed_rows)} rows  |  "
            f"file: {os.path.basename(self.data_file)}"
        )

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Load labelled CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._load_dataset_file(path)

    def _on_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save dataset as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.data_file = os.path.abspath(path)
            self.data_path_var.set(self.data_file)
            self.dataset.save(self.data_file)
            self._update_dataset_label()
            self._set_status(f"Saved {len(self.dataset)} rows to {self.data_file}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _training_set(self) -> Dataset:
        """The dataset actually used to train the model.

        The user's own rows are always included.  The built-in seed corpus is
        added as a baseline so that a model trained on only a handful of the
        user's grades still behaves sensibly instead of collapsing to a
        single class.  The seed rows are never written to the user's CSV, and
        because training is a plain full refit they are treated exactly like
        any other row.
        """
        return build_training_set(self.dataset.examples, self.seed_rows)

    def _retrain(self, quiet: bool = False) -> None:
        """Full refit on the current dataset."""
        training = self._training_set()
        if len(training) == 0:
            self._set_status("Nothing to train on - dataset is empty")
            return
        try:
            self.classifier.fit(training)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status(f"Training failed: {exc}")
            if not quiet:
                messagebox.showerror("Training error", str(exc))
            return
        self._refresh_model_panel()
        self._rescore_candidates()
        if not quiet:
            self._set_status(
                f"Retrained (full refit) on {len(training)} rows "
                f"({len(self.dataset)} yours + {len(self.seed_rows)} seed) - "
                f"CV accuracy {self._cv_text()}"
            )

    def _refresh_model_panel(self) -> None:
        info = self.classifier.info
        self.model_var.set(
            f"Model: random forest ({info.n_samples} samples, "
            f"{info.n_positive} pass / {info.n_negative} fail)\n"
            f"CV accuracy: {self._cv_text()} - {info.cv_note}"
        )
        self.importance_text.delete("1.0", "end")
        for name, value in self.classifier.feature_importances(top=12):
            self.importance_text.insert("end", f"{value:6.3f}  {name}\n")
        self.detail_var.set(
            f"Trained: {info.trained_at}  |  {info.feature_count} features  |  "
            f"params: {info.params}"
        )

    def _cv_text(self) -> str:
        acc = self.classifier.info.cv_accuracy
        return f"{acc:.1%}" if acc is not None else "n/a"

    def _on_save_model(self) -> None:
        if not self.classifier.is_trained():
            messagebox.showinfo("No model", "Train a model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".joblib",
            initialfile=DEFAULT_MODEL_FILE,
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
        )
        if path:
            try:
                saved = save_model(self.classifier, path)
                self._set_status(f"Model saved to {saved}")
            except Exception as exc:
                messagebox.showerror("Save error", str(exc))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _rebuild_generator(self) -> None:
        # Markov mode learns from the user's own pass examples, falling back to
        # the seed positives until the user has graded enough of their own.
        positives = [e.name for e in self.dataset.positives()]
        if len(positives) < 2:
            positives += [n for n, label in self.seed_rows if label == 1]
        self.generator = NameGenerator(
            corpus_words=self.corpus_words, positives=positives
        )

    def _mode(self) -> str:
        return MODE_VALUES.get(self.mode_var.get(), "random")

    def _on_mode_change(self) -> None:
        self._set_status(f"Mode: {self.mode_var.get()}")
        if self._mode() == "markov":
            pos = len(self.dataset.positives())
            if pos < 2:
                self._set_status(
                    "Markov mode needs at least 2 names marked as pass "
                    f"(currently {pos}). It will fall back to the corpus."
                )

    def _threshold(self) -> float:
        """The pass threshold, clamped to a sane range."""
        try:
            value = float(self.threshold_var.get())
        except (tk.TclError, ValueError):
            value = DEFAULT_THRESHOLD
        if value < 0.01:
            value = 0.01
        if value > 0.99:
            value = 0.99
        return value

    def _lengths(self) -> Tuple[int, int]:
        try:
            lo = int(self.min_len_var.get())
            hi = int(self.max_len_var.get())
        except (tk.TclError, ValueError):
            lo, hi = 4, 10
        if lo < 2:
            lo = 2
        if hi > 24:
            hi = 24
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    def _on_generate_one(self) -> None:
        self._generate(1)

    def _on_generate_ten(self) -> None:
        self._generate(10)

    def _on_generate_accepted(self) -> None:
        """Generate until the classifier accepts 5 candidates."""
        if not self.classifier.is_trained():
            messagebox.showinfo("No model", "Train a model first (Retrain).")
            return
        lo, hi = self._lengths()
        wanted = 5
        threshold = self._threshold()
        accepted = self.generator.generate_until(
            scorer=self.classifier.score,
            mode=self._mode(),
            wanted=wanted,
            min_length=lo,
            max_length=hi,
            threshold=threshold,
        )
        if not accepted:
            self._set_status(
                "No candidate passed the threshold. Try lowering it, widening the "
                "length range, or retraining."
            )
            return
        names = [n for n, _ in accepted]
        graded = self.classifier.grade(names, threshold=threshold)
        self._set_candidates(graded)
        self._set_status(f"Generated {len(graded)} accepted candidate(s).")

    def _generate(self, count: int) -> None:
        if self.generator is None:
            self._rebuild_generator()
        lo, hi = self._lengths()
        mode = self._mode()
        if mode == "accepted":
            self._on_generate_accepted()
            return
        names = self.generator.generate(
            mode=mode, count=count, min_length=lo, max_length=hi
        )
        if not names:
            self._set_status("Could not generate a name with those constraints.")
            return
        graded = self.classifier.grade(names, threshold=self._threshold())
        self._set_candidates(graded)
        self._set_status(f"Generated {len(graded)} candidate(s).")

    def _set_candidates(self, graded: List[Tuple[str, float, bool]]) -> None:
        self.candidates = list(graded)
        self.tree.delete(*self.tree.get_children())
        for name, score, passes in self.candidates:
            verdict = "PASS" if passes else "FAIL"
            self.tree.insert(
                "", "end", values=(name, f"{score:.3f}", verdict)
            )
        if self.candidates:
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)

    def _rescore_candidates(self) -> None:
        """Re-grade the current candidates after a retrain or threshold change."""
        if not self.candidates:
            return
        names = [n for n, _, _ in self.candidates]
        graded = self.classifier.grade(names, threshold=self._threshold())
        self.candidates = list(graded)
        by_name = {gname: (score, passes) for gname, score, passes in graded}
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            name = values[0]
            if name in by_name:
                score, passes = by_name[name]
                self.tree.item(
                    item, values=(name, f"{score:.3f}", "PASS" if passes else "FAIL")
                )

    def _on_threshold_change(self) -> None:
        self._rescore_candidates()

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    def _on_select(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0], "values")
        self.selected_var.set(values[0] if values else "(none)")

    def _on_grade(self, label: int) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Nothing selected", "Select a candidate first.")
            return
        values = self.tree.item(selection[0], "values")
        name = values[0]
        try:
            path = append_example(self.data_file, name, label)
        except (DatasetError, OSError) as exc:
            messagebox.showerror("Could not save", str(exc))
            return

        # Keep the in-memory dataset in sync with what is on disk.
        self.dataset.add(name, label)
        self.data_file = path
        self.data_path_var.set(path)
        self._update_dataset_label()
        self._rebuild_generator()

        verdict = "pass" if label == 1 else "fail"
        self._set_status(
            f"Recorded '{name}' as {verdict} -> {os.path.basename(path)}. "
            "Click Retrain to apply it (full refit)."
        )
        # Mark the row so the user can see what has been graded.
        self.tree.set(selection[0], "verdict", "PASS" if label == 1 else "FAIL")
        self.tree.set(selection[0], "score", "graded")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def run(self) -> None:
        self.root.mainloop()


def launch(
    data_file: str = DEFAULT_DATA_FILE, corpus: Optional[List[str]] = None
) -> None:
    """Start the GUI.

    Args:
        data_file: path to the labelled CSV to use.
        corpus: optional word list to use as the generation corpus.  When
            ``None`` the built-in word list in :mod:`seed_data` is used.
    """
    root = tk.Tk()
    app = NameGeneratorApp(root, data_file=data_file)
    if corpus:
        app.corpus_words = list(corpus)
        app._rebuild_generator()
        app._set_status(f"Using a custom corpus of {len(corpus)} words.")
    root.mainloop()
