"""Aggregate + plot the stem-final-match / conjugation / L-NL probe results.

Reads the per-(arch, run) CSVs written by run_probes_stemfinal_lnl.py and
writes summary_<split>.csv (mean/std across runs) plus accuracy-by-layer
trajectory plots, one panel per property, with baselines marked.

Usage:
  summarize_stemfinal_lnl.py [--results-dir DIR] [--split SPLIT]
                             [--probe-type TYPE] [--baselines-dir DIR]
  summarize_stemfinal_lnl.py (-h | --help)

Options:
  --results-dir DIR    Per-run probe CSVs [default: data/probing/results_stemfinal_lnl_grouped].
  --split SPLIT        Data split [default: 10L_90NL].
  --probe-type TYPE    Probe drawn in the trajectory plot, linear or mlp;
                       both go in the CSV [default: linear].
  --baselines-dir DIR  Surface n-gram baseline results (run_ngram_baselines.py);
                       overlaid as horizontal lines and appended to the summary
                       CSV. Silently skipped when absent
                       [default: data/probing/results_ngram_baselines_balanced].
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from probing import MODEL_TYPES
from probing.utils.cli import parse

PROPERTIES = ("stem_final_match", "conjugation", "l_shaped")
PROPERTY_TITLES = {
    "stem_final_match": "stem-final match\n(same vs differ, per instance)",
    "conjugation": "conjugation class\n(-ar / -er / -ir)",
    "l_shaped": "L-shaped membership\n(L vs NL)",
}


def layer_key(row):
    """Sortable index: encoder layers first (0..), then decoder layers."""
    base = 0 if row["layer_type"] == "encoder" else 100
    return base + int(row["layer_index"])


def layer_label(row):
    return f"{row['layer_type'][:3]}{row['layer_index']}"


def load_all(results_dir):
    frames = []
    for arch in MODEL_TYPES:
        # A run's CSV is variant-tagged when produced under non-default flags
        # (e.g. *_results.control.csv from --control). Prefer the control
        # variant (superset schema) over the plain one for the same run.
        by_run = {}
        for f in sorted(
            glob.glob(os.path.join(results_dir, arch, "*_stemfinal_lnl_results.csv"))
            + glob.glob(os.path.join(results_dir, arch, "*_stemfinal_lnl_results.control.csv"))
        ):
            run_id = os.path.basename(f).split("_stemfinal_lnl_results")[0]
            if f.endswith("_results.control.csv") or run_id not in by_run:
                by_run[run_id] = f
        for run_id in sorted(by_run):
            df = pd.read_csv(by_run[run_id])
            df["arch"] = arch
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No result CSVs found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_baselines(baselines_dir):
    """Load surface n-gram baseline summary rows, or None (layer_index is the n-gram order)."""
    frames = []
    for f in sorted(glob.glob(os.path.join(baselines_dir, "*", "*_stemfinal_lnl_results.csv"))):
        df = pd.read_csv(f)
        df["arch"] = os.path.basename(os.path.dirname(f))
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    grp = df.groupby(["arch", "property", "probe_site", "layer_index", "probe_type"], as_index=False)
    summary = grp.agg(
        n_runs=("accuracy", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        majority_baseline_mean=("majority_baseline", "mean"),
        majority_baseline_std=("majority_baseline", "std"),
    )
    summary["lkey"] = 200 + summary["layer_index"]
    summary["layer"] = summary["layer_index"].map(
        lambda n: "inf-gram" if n < 0 else f"ngram{int(n)}"
    )
    return summary.sort_values(["arch", "property", "probe_type", "probe_site", "lkey"])


def parse_args():
    return parse(__doc__, choices=dict(probe_type=("linear", "mlp")))


if __name__ == "__main__":
    args = parse_args()

    df = load_all(args.results_dir)
    df["lkey"] = df.apply(layer_key, axis=1)
    df["layer"] = df.apply(layer_label, axis=1)

    # 1. summary CSV: mean/std over runs
    metrics = ["accuracy", "majority_baseline"]
    for extra in ("control_accuracy", "selectivity"):
        if extra in df.columns:
            metrics.append(extra)
    grp = df.groupby(["arch", "property", "lkey", "layer", "probe_type"], as_index=False)
    summary = grp.agg(
        n_runs=("accuracy", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).sort_values(["arch", "property", "probe_type", "lkey"])
    baselines = load_baselines(args.baselines_dir)
    summary_path = os.path.join(args.results_dir, f"summary_{args.split}.csv")
    out = summary if baselines is None else pd.concat([summary, baselines], ignore_index=True)
    out.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}  ({len(out)} rows)")

    # 2. trajectory plot
    sub = summary[summary["probe_type"] == args.probe_type]
    fig, axes = plt.subplots(1, len(PROPERTIES), figsize=(5.2 * len(PROPERTIES), 4.6), sharey=True)
    cmap = plt.get_cmap("tab10")
    archs = [a for a in MODEL_TYPES if a in sub["arch"].unique()]

    for ax, prop in zip(axes, PROPERTIES):
        s = sub[sub["property"] == prop]
        order = s.sort_values("lkey")["layer"].drop_duplicates().tolist()
        x = range(len(order))
        for i, arch in enumerate(archs):
            a = s[s["arch"] == arch].set_index("layer").reindex(order)
            y = a["accuracy_mean"].values
            e = a["accuracy_std"].fillna(0).values
            ax.plot(x, y, marker="o", color=cmap(i), label=arch, linewidth=1.8)
            ax.fill_between(x, y - e, y + e, color=cmap(i), alpha=0.12)
        base = s["majority_baseline_mean"].mean()
        ax.axhline(base, ls=":", color="black", lw=1, label=f"maj. baseline ({base:.2f})")
        if baselines is not None:
            b = baselines[baselines["property"] == prop]
            for ptype, ls, color, name in (
                ("ngram-linear", "--", "dimgray", "ngram clf"),
                ("ngram-lm", "-.", "darkred", "ngram LM"),
                ("infinigram", (0, (1, 1)), "purple", "inf-gram LM"),
            ):
                bb = b[b["probe_type"] == ptype]
                if bb.empty:
                    continue
                best = bb.loc[bb["accuracy_mean"].idxmax()]
                n_lab = "inf" if best["layer_index"] < 0 else int(best["layer_index"])
                ax.axhline(
                    best["accuracy_mean"], ls=ls, color=color, lw=1.4,
                    label=f"{name} ({best['probe_site'][:3]},n{n_lab}) {best['accuracy_mean']:.2f}",
                )
        ax.set_xticks(list(x))
        ax.set_xticklabels(order, rotation=45)
        ax.set_title(PROPERTY_TITLES[prop], fontsize=10)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"accuracy ({args.probe_type} probe, lemma-disjoint CV)")
    axes[0].set_ylim(0.4, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(archs) + 1,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Stem-final / conjugation / L-NL decodability by layer — {args.split}, mean±run-std over 3 test sets",
        y=1.08, fontsize=11,
    )
    fig.tight_layout()
    png_path = os.path.join(args.results_dir, f"trajectory_{args.split}_{args.probe_type}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {png_path}")
