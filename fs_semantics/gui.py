from __future__ import annotations

from .decomposition_service import generate_decomposition_bundle
from .example_generator import generate_example_from_sequent


def launch_gui() -> int:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import scrolledtext
    from tkinter import ttk

    root = tk.Tk()

    root.title("Fractional Semantics Decomposition")
    root.geometry("1100x760")

    container = ttk.Frame(root, padding=12)
    container.pack(fill="both", expand=True)

    input_frame = ttk.LabelFrame(container, text="Inputs", padding=10)
    input_frame.pack(fill="x")

    ttk.Label(input_frame, text="Sequent").grid(
        row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 6)
    )
    sequent_var = tk.StringVar()
    sequent_entry = ttk.Entry(input_frame, textvariable=sequent_var, width=120)
    sequent_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))

    ttk.Label(
        input_frame,
        text="Belief set B (optional)",
    ).grid(row=1, column=0, sticky="nw", padx=(0, 10))
    belief_text = tk.Text(input_frame, height=4, width=90, wrap="word")
    belief_text.grid(row=1, column=1, sticky="ew")

    hint = (
        'Accepted formats: "B={(p | q); ~u}" or "(p | q); ~u" '
        "(semicolon-separated, one line also works)."
    )
    ttk.Label(input_frame, text=hint).grid(
        row=2, column=1, sticky="w", pady=(6, 0)
    )
    ttk.Label(
        input_frame,
        text=(
            "Legend: OR = |, ∨, \\vee, OR | "
            "AND = &, ∧, \\wedge, AND | "
            "NEG = ~, !, ¬, \\neg, \\overline{p}, NOT | "
            "precedence: NOT > AND > OR, so u | p & v = u | (p & v)"
        ),
    ).grid(row=3, column=1, sticky="w", pady=(4, 0))
    ttk.Label(
        input_frame,
        text="Example mode",
    ).grid(row=4, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
    example_mode = ttk.Combobox(
        input_frame,
        values=(
            "auto",
            "local_only",
            "template_only",
        ),
        state="readonly",
        width=24,
    )
    example_mode.set("auto")
    example_mode.grid(row=4, column=1, sticky="w", pady=(6, 0))

    input_frame.columnconfigure(1, weight=1)

    button_frame = ttk.Frame(container)
    button_frame.pack(fill="x", pady=(10, 8))

    output_frame = ttk.LabelFrame(
        container,
        text="Decomposition (only)",
        padding=10,
    )
    output_frame.pack(fill="both", expand=True)

    output = scrolledtext.ScrolledText(
        output_frame,
        wrap="none",
        font=("Courier", 11),
    )
    output.pack(fill="both", expand=True)

    def _clear() -> None:
        sequent_var.set("")
        belief_text.delete("1.0", tk.END)
        output.delete("1.0", tk.END)

    def _generate() -> None:
        sequent_text = sequent_var.get().strip()
        if not sequent_text:
            messagebox.showerror("Input error", "Please provide a sequent.")
            return

        belief_set_raw = belief_text.get("1.0", tk.END).strip()

        try:
            bundle = generate_decomposition_bundle(
                sequent_text,
                belief_set_raw,
                decomposition_style="cut_free",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Computation error", str(exc))
            return

        decomposition = str(bundle["decomposition"])
        equivalence = bundle.get("equivalence")
        prefix = ""
        if isinstance(equivalence, dict) and bool(equivalence.get("changed")):
            chain = str(equivalence.get("ascii_chain", ""))
            note = str(equivalence.get("note", ""))
            prefix = "Equivalence chain: " + chain
            if note:
                prefix += f"\n{note}"
            prefix += "\n\n"

        output.delete("1.0", tk.END)
        output.insert("1.0", prefix + decomposition)

    def _generate_example() -> None:
        sequent_text = sequent_var.get().strip()
        if not sequent_text:
            messagebox.showerror("Input error", "Please provide a sequent.")
            return
        belief_set_raw = belief_text.get("1.0", tk.END).strip()
        try:
            example = generate_example_from_sequent(
                sequent_text,
                belief_set_text=belief_set_raw,
                prefer_local_model=True,
                generation_mode=str(example_mode.get() or "auto"),
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Example generation error", str(exc))
            return

        output.delete("1.0", tk.END)
        output.insert(
            "1.0",
            (
                f"Example generator engine: {example.engine} "
                f"(model: {example.model})\n\n{example.text}"
            ),
        )

    ttk.Button(
        button_frame,
        text="Generate Decomposition",
        command=_generate,
    ).pack(side="left")
    ttk.Button(
        button_frame,
        text="Generate Example",
        command=_generate_example,
    ).pack(side="left", padx=(8, 0))
    ttk.Button(button_frame, text="Clear", command=_clear).pack(side="left", padx=(8, 0))

    sequent_entry.focus_set()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_gui())
